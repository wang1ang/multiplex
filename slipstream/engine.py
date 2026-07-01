"""L1 — engine layer.

The lowest layer. Its only job: run correct batched forward passes and manage
the batched cache. It does NOT sample, does NOT decide AR-vs-MTP, does NOT know
about requests. It gives whatever it's fed a correct forward.

Two capabilities, nothing more:
  * batch: process B sequences together in one forward.
  * next-k: feed k tokens per row at once and return all k positions' logits
    (AR is just k=1; MTP verify is k>1 — the engine doesn't care which).

Correctness facts (verified by experiment):
  * Batched forward of this hybrid SSM+attention model is numerically EXACT vs
    single-sequence when rows are equal length. Verified token-for-token.
  * Unequal-length prefill is handled by right-padding + masking. An mlx-lm bug
    left SSM padding unmasked (see prefill() for the fix); with the fix, the pad
    positions are correctly masked and do NOT corrupt the recurrent state.
  * Residual divergence between a batched row and the same sequence run alone is
    pure floating-point accumulation (batch reduction order differs from B=1).
    This is inherent to batched inference — NOT a bug — and both trajectories are
    valid samples. It shows up only after many steps as an occasional token flip.
  * The cache is ours to roll back for speculative verify: ``snapshot_ssm`` /
    ``restore_ssm`` save & restore SSM recurrent state (it can't be trimmed),
    and ``trim_attention`` trims attention KV. Together they undo a rejected
    verify forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import _make_cache, _right_pad_prompts
from mlx_lm.models.cache import ArraysCache


@dataclass
class BatchState:
    """Batched cache + per-row bookkeeping for B aligned sequences."""

    cache: list[Any]        # batched per-layer caches (BatchKVCache / ArraysCache)
    lengths: list[int]      # committed token count per row (prompt + accepted)

    @property
    def batch_size(self) -> int:
        return len(self.lengths)


class Engine:
    """Loads a model. Runs correct batched forwards. Nothing else."""

    def __init__(self, model_path: str):
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        self.model_path = model_path
        self.load_seconds = time.time() - t0

    # --- tokenization ---
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        # skip_special_tokens drops <|im_end|>/<|endoftext|> etc from the text.
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_ids(self) -> set[int]:
        return set(self.tokenizer.eos_token_ids)

    def logits(self, hidden: mx.array) -> mx.array:
        """Trunk lm_head over hidden -> logits ``[..., vocab]``."""
        return self.model.language_model.lm_head(hidden)

    # --- batched forward primitives (always [B, ...]; B=1 is just a batch of 1) ---
    def prefill(self, prompts: list[list[int]]) -> tuple[BatchState, mx.array]:
        """Prefill B prompts. Returns (state, hidden ``[B, max_len, H]``).

        Row i's next-token hidden is at position ``lengths[i]-1`` (prompts are
        right-padded to max_len). Returns pre-lm_head hidden; get logits with
        ``logits(hidden)``. The caller samples — the engine doesn't.
        """
        lengths = [len(p) for p in prompts]
        max_len = max(lengths)
        padding = [max_len - n for n in lengths]
        cache = _make_cache(self.model, [0] * len(prompts), None)
        tokens = _right_pad_prompts(prompts, max_length=max_len)
        for c in cache:
            c.prepare(lengths=lengths, right_padding=padding)
            # mlx-lm bug: _make_cache sets ArraysCache.left_padding = [0,...],
            # so make_mask() takes the left_padding branch (pos >= 0, always True)
            # and never masks right-padding — pad tokens corrupt GatedDeltaNet
            # state. Clearing it forces the lengths branch (pos < lengths), which
            # masks the pad positions. (padding=0 -> masks nothing, still correct.)
            if isinstance(c, ArraysCache):
                c.left_padding = None
        h = self.model.language_model.model(tokens, cache=cache)
        for c in cache:
            c.finalize()
        return BatchState(cache=cache, lengths=list(lengths)), h

    def forward(self, state: BatchState, tokens: mx.array) -> mx.array:
        """Feed ``tokens`` (``[B, k]``) per row, return hidden ``[B, k, H]``.

        k=1 is AR; k>1 is speculative verify. Returns ALL k positions' hidden
        (pre-lm_head; use ``logits()``); does not slice or sample. Advances the
        cache by k; use snapshot/trim to discard rejected positions.
        """
        k = int(tokens.shape[1])
        h = self.model.language_model.model(tokens, cache=state.cache)
        state.lengths = [n + k for n in state.lengths]
        return h

    def snapshot_ssm(self, state: BatchState) -> list:
        """Clone the SSM (ArraysCache) recurrent state of every SSM layer.

        SSM state can't be trimmed (it evolves sequentially), so speculative
        verify saves it here and restores after. Attention layers are skipped
        (they trim instead). The clone forces evaluation off the lazy graph
        (``v + 0``) so later cache writes don't mutate the snapshot.
        """
        snap = []
        for c in state.cache:
            if isinstance(c, ArraysCache):
                snap.append([None if v is None else v + 0 for v in c.cache])
            else:
                snap.append(None)
        return snap

    def restore_ssm(self, state: BatchState, snap: list) -> None:
        """Write a snapshot_ssm() result back into the SSM layers."""
        for c, s in zip(state.cache, snap):
            if s is not None:
                c.cache = [None if v is None else v + 0 for v in s]

    def trim_attention(self, state: BatchState, n: int) -> None:
        """Trim n positions off every attention (KVCache) layer. SSM layers are
        left untouched — restore them with restore_ssm()."""
        if n <= 0:
            return
        for c in state.cache:
            if not isinstance(c, ArraysCache):
                c.trim(n)

    def filter(self, state: BatchState, keep: list[int]) -> None:
        """Keep only rows ``keep`` (by row index) in the batched cache; drop the
        rest. Used to remove finished (EOS) rows so the batch shrinks. Every cache
        layer (BatchKVCache / ArraysCache) supports filter(indices)."""
        for c in state.cache:
            c.filter(keep)
        state.lengths = [state.lengths[i] for i in keep]
