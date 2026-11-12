"""DFlash 2 drafter components (Inco AI, checkpoints ``incoai/*-DFlash2``, Apache-2.0).

Extracted unmodified from mlx-dspark 0.20.4 ``dflash_model.py`` (MIT, see LICENSE):
the two trained components DFlash 2 adds on top of the DFlash 1 backbone. They depend
on nothing but ``mlx``; ``kernel/dflash.py`` wires them into its draft model.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class DFlashGroupedConv(nn.Module):
    """DFlash 2: grouped dynamic depthwise K-tap convolution across one draft block.

    Wraps one sublayer: ``prepare`` convolves its input and returns the coefficient set for
    ``finish`` to convolve its output — BOTH sets come from one projection of the input.
    Each coefficient is a learned per-channel base plus a content-adaptive per-group
    correction (``group_size`` channels share one correction). Applied only to the block
    rows (the ctx path never sees it), so the zero-padded tap shift is exactly the
    reference's block-boundary mask: position 0 (the anchor) has no in-block predecessor,
    position 1 reads the anchor's representation.
    """

    def __init__(self, hidden_size: int, taps: int, group_size: int):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"conv_group_size={group_size} must divide hidden_size={hidden_size}")
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        # [side (in/out), tap, channel] — identity at init (tap 0 = 1), the layout training exports
        self.base_kernel = mx.concatenate(
            [mx.ones((2, 1, hidden_size)), mx.zeros((2, taps - 1, hidden_size))], axis=1)
        self.kernel_projection = nn.Linear(hidden_size, 2 * taps * self.num_groups, bias=False)

    def _convolve(self, x, delta, side: int):
        # x [B, L, H]; delta [B, L, taps, num_groups]
        B, L, H = x.shape
        xg = x.reshape(B, L, self.num_groups, self.group_size)
        coeff = (self.base_kernel[side].reshape(1, 1, self.taps, self.num_groups, self.group_size)
                 + delta[..., None])
        out = coeff[:, :, 0] * xg
        for t in range(1, self.taps):
            shifted = mx.pad(xg[:, :-t], ((0, 0), (t, 0), (0, 0), (0, 0)))
            out = out + coeff[:, :, t] * shifted
        return out.reshape(B, L, H)

    def prepare(self, x):
        coeff = self.kernel_projection(x).reshape(*x.shape[:-1], 2, self.taps, self.num_groups)
        return self._convolve(x, coeff[..., 0, :, :], 0), coeff[..., 1, :, :]

    def finish(self, y, delta):
        return self._convolve(y, delta, 1)


class CandidateSelector(nn.Module):
    """DFlash 2: scores the K x K transitions between adjacent mask slots, then walks them.

    ``lattice`` builds ``scores[slot, p, c] = unary[slot, c] + <A[pred] * proj(h_slot), B[c]>``
    where the predecessors of slot 0 are all the verified anchor token and of slot s are
    slot s-1's candidates. ``walk_greedy`` follows the best successor from the anchor;
    ``walk_sampled`` draws the walk by inverse CDF (one uniform per slot, shared across
    predecessor rows, matching the reference) and returns the realized q rows over the K
    candidates for the spec-sampling accept. Scores are fp32 like the reference (bf16
    bilinear + fp32 unary).
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = mx.zeros((vocab_size, rank))
        self.successor_codebook = mx.zeros((vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def lattice(self, candidate_ids, unary_logits, hidden, anchor_id: int):
        # candidate_ids [g, K] int; unary_logits [g, K] fp32; hidden [g, H] (post-final-norm)
        k = candidate_ids.shape[1]
        h = self.hidden_projection(hidden)                                    # [g, r]
        anchor = mx.full((1, k), anchor_id, dtype=candidate_ids.dtype)
        pred_ids = mx.concatenate([anchor, candidate_ids[:-1]], axis=0)       # [g, K]
        pre = self.predecessor_codebook[pred_ids] * h[:, None, :]             # [g, K, r]
        suc = self.successor_codebook[candidate_ids]                          # [g, K, r]
        bilinear = pre @ suc.transpose(0, 2, 1)                               # [g, Kpred, Kcand]
        return unary_logits[:, None, :].astype(mx.float32) + bilinear.astype(mx.float32)

    def walk_greedy(self, scores, candidate_ids):
        """Chain-argmax path -> drafted token ids [g], fully in-graph (no sync)."""
        g = scores.shape[0]
        idx = mx.argmax(scores[0, 0])           # slot 0: every predecessor row is the anchor
        picks = [candidate_ids[0, idx]]
        for s in range(1, g):
            idx = mx.argmax(scores[s, idx])
            picks.append(candidate_ids[s, idx])
        return mx.stack(picks)

    def walk_sampled(self, scores, candidate_ids, uniforms, temperature: float):
        """Sampled walk -> (drafted ids [g], q rows [g, K] the tokens were drawn from)."""
        g, k = candidate_ids.shape
        t = max(float(temperature), 1e-5)
        init = mx.softmax(scores[0, 0] / t, axis=-1)                          # [K]
        idx = mx.minimum((uniforms[0] >= mx.cumsum(init)).sum(), k - 1)
        picks, q_rows = [candidate_ids[0, idx]], [init]
        if g > 1:
            trans = mx.softmax(scores[1:] / t, axis=-1)                       # [g-1, K, K]
            maps = mx.minimum(
                (uniforms[1:, None, None] >= mx.cumsum(trans, axis=-1)).sum(axis=-1), k - 1)
            for s in range(1, g):
                q_rows.append(trans[s - 1, idx])     # realized row BEFORE stepping
                idx = maps[s - 1, idx]
                picks.append(candidate_ids[s, idx])
        return mx.stack(picks), mx.stack(q_rows)

