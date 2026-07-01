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
  * The cache is ours to trim: ``rollback(state, keep)`` shrinks each row's
    cache. Attention layers trim; SSM layers can't yet (see rollback()) — a
    real MTP verify path will need SSM-state rollback too.
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
        return self.tokenizer.decode(token_ids)

    @property
    def eos_token_ids(self) -> set[int]:
        return set(self.tokenizer.eos_token_ids)

    # --- batched forward primitives ---
    def prefill(self, prompts: list[list[int]]) -> tuple[BatchState, mx.array]:
        """Prefill B prompts. Returns (state, logits).

        ``logits`` is ``[B, max_len, vocab]``; row i's next-token logits are at
        position ``lengths[i]-1`` (prompts are right-padded to max_len). The
        caller samples — the engine doesn't.
        """
        B = len(prompts)
        lengths = [len(p) for p in prompts]
        max_len = max(lengths)
        padding = [max_len - n for n in lengths]

        cache = _make_cache(self.model, [0] * B, None)
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

        logits = self.model(tokens, cache=cache)

        for c in cache:
            c.finalize()

        state = BatchState(cache=cache, lengths=list(lengths))
        return state, logits

    def forward(self, state: BatchState, tokens: mx.array) -> mx.array:
        """Feed ``tokens`` (``[B, k]``) for every row and return ``[B, k, vocab]``.

        AR is k=1; speculative verify is k>1. The engine returns ALL k positions'
        logits — it does not slice to the last one, does not sample. The cache
        advances by k for every row; use ``rollback`` afterwards to discard
        positions the caller rejected.
        """
        k = int(tokens.shape[1])
        logits = self.model(tokens, cache=state.cache)
        state.lengths = [n + k for n in state.lengths]
        return logits

    def rollback(self, state: BatchState, keep: list[int]) -> None:
        """Drop the last tokens so row i keeps ``keep[i]`` positions.

        Discards rejected speculative tokens. All rows drop the same count (the
        batched cache trims uniformly) — the take-min speculative scheme
        guarantees it, since all rows commit the min accepted length.

        SSM (ArraysCache) recurrent-state rollback is not implemented yet, so
        this raises if the cache has SSM layers. It must be solved before MTP
        verify can use rollback.
        """
        drop = [old - new for old, new in zip(state.lengths, keep)]
        if len(set(drop)) != 1:
            raise ValueError(f"rollback needs equal drop per row, got {drop}")
        n = drop[0]
        if n <= 0:
            return
        for c in state.cache:
            if isinstance(c, ArraysCache):
                raise NotImplementedError("SSM-layer rollback not implemented")
            c.trim(n)
        state.lengths = list(keep)
