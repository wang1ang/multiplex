"""L3 prefix-cache state adapter.

This module owns the model-cache mechanics needed by prefix reuse:
extracting attention KV block deltas, snapshotting SSM boundary state, and
rebuilding a single-row BatchState from a policy match. The engine stays focused
on forward/batch primitives; policy stays model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from ..engine import BatchState, concat_kv, make_batch_cache, row_kv

if TYPE_CHECKING:
    from ..engine import Engine
    from .policy import Match


COMBINED_BLOCK_VERSION = 1


@dataclass
class RestoredPrefix:
    state: BatchState
    pos: int
    cached_h: Any | None = None
    mtp_blocks: list[Any] | None = None


class PrefixCacheState:
    """Bridge between L3 prefix-cache policy and MLX cache objects."""

    def __init__(self, engine: "Engine"):
        self.engine = engine

    def clone_attention_block(self, state: BatchState, start: int, pos: int):
        """Clone attention-only KV for ``[start:pos]`` from a single-row state."""
        start = int(start)
        pos = int(pos)
        block = []
        leaves = []
        for c in state.cache:
            if isinstance(c, ArraysCache):
                block.append(None)
                continue
            keys, values, length = row_kv(c)
            if start < 0 or pos > length or pos <= start:
                raise ValueError(
                    f"invalid attention block slice [{start}:{pos}] for length={length}"
                )
            k = keys[..., start:pos, :] + 0
            v = values[..., start:pos, :] + 0
            block.append([k, v])
            leaves.extend([k, v])
        if leaves:
            mx.eval(*leaves)
        return block

    def clone_ssm(self, cache: list[Any]):
        """Clone only SSM/ArraysCache state at a reusable boundary."""
        snap = []
        leaves = []
        for c in cache:
            if isinstance(c, ArraysCache):
                layer = [None if v is None else v + 0 for v in c.cache]
                leaves.extend(v for v in layer if v is not None)
                snap.append(layer)
            else:
                snap.append(None)
        if leaves:
            mx.eval(*leaves)
        return snap

    @staticmethod
    def pack_cache_block(attn, mtp=None):
        # Policy/disk see one opaque block payload; trunk and MTP deltas must
        # move together so restore cannot warm one cache and cold-start the other.
        if mtp is None:
            return attn
        return (COMBINED_BLOCK_VERSION, attn, mtp)

    def split_cache_blocks(self, blocks: list[Any]) -> tuple[list[Any], list[Any] | None]:
        attn_blocks = []
        mtp_blocks: list[Any] | None = []
        for block in blocks:
            if (
                isinstance(block, tuple)
                and len(block) == 3
                and block[0] == COMBINED_BLOCK_VERSION
            ):
                attn_blocks.append(block[1])
                if mtp_blocks is not None:
                    mtp_blocks.append(block[2])
            else:
                attn_blocks.append(block)
                mtp_blocks = None
        return attn_blocks, mtp_blocks

    def restore_match(self, match: "Match") -> RestoredPrefix:
        """Restore the block-tree payload returned by policy.find()."""
        payload = match.payload
        kind = payload[0] if payload else None
        if kind != "blocks":
            raise ValueError(f"unsupported prefix-cache payload: {kind!r}")
        _kind, blocks, ssm, pos, cached_h = payload
        attn_blocks, mtp_blocks = self.split_cache_blocks(blocks)
        state = self.restore_blocks(attn_blocks, ssm, pos)
        return RestoredPrefix(
            state=state,
            pos=int(pos),
            cached_h=cached_h,
            mtp_blocks=mtp_blocks,
        )

    def restore_blocks(self, blocks: list[Any], ssm_snapshot, pos: int) -> BatchState:
        """Rebuild state by concatenating attention block deltas and applying SSM."""
        cache = make_batch_cache(self.engine.model, [0])
        for layer_idx, c in enumerate(cache):
            if isinstance(c, ArraysCache):
                layer_ssm = ssm_snapshot[layer_idx]
                c.cache = [None if v is None else v + 0 for v in layer_ssm]
                continue

            parts = [block[layer_idx] for block in blocks if block[layer_idx] is not None]
            if not parts:
                continue
            keys, values = concat_kv(parts)
            mx.eval(keys, values)
            self._assign_attention(c, keys, values)
        return BatchState(cache=cache, lengths=[int(pos)])

    @staticmethod
    def _assign_attention(c, keys, values) -> None:
        if isinstance(c, BatchKVCache):
            c.keys, c.values = keys, values
            c.offset = mx.array([keys.shape[2]])
            c.left_padding = mx.array([0])
            c._idx = keys.shape[2]
        else:
            c.state = [keys, values]
