"""Prefix-cache policy: chunk-chain longest-prefix matching + per-pool LRU.

This module has no model knowledge. It only tracks token prefixes and opaque
payloads supplied by the L3 state adapter.

The core invariant is:
  * prefill is chunk-aligned, so every stored block spans exactly
    ``[i * chunk, (i + 1) * chunk)`` and the tree of blocks is a chain of
    fixed-size chunks;
  * a block is identified by ``key = H(parent_key, chunk_tokens)``, so the
    deepest reusable prefix is found by rehashing the request chunk by chunk
    instead of indexing individual tokens;
  * attention KV is stored as per-block deltas on those chain nodes;
  * SSM is stored on reusable boundary nodes;
  * a match restores the parent-chain attention blocks plus that node's SSM.

When ``disk_dir`` is set, block records are written in the background. Startup
restores only metadata and loads tensor blobs lazily on cache hits.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterator

from .disk import AsyncPrefixDiskStore, DiskBlockRecord, block_key


ROOT_KEY = ""

@dataclass
class Node:
    """One chunk-aligned block in the prefix chain."""

    key: str
    pos: int
    parent: "Node | None" = None
    attn: Any | None = None
    ssm: Any | None = None
    cached_h: Any | None = None
    source: str | None = None
    pool: str = "default"
    touch: int = 0
    reusable: bool = False


@dataclass
class Match:
    prefix_len: int
    payload: Any
    source: str | None = None
    pool: str = "default"
    key: str | None = None


class PrefixCache:
    """Chunk-chain prefix cache with independent per-pool LRU.

    ``capacity`` bounds resident reusable payloads per pool.

    Prefix-chain nodes backed by a disk record are retained after their tensors
    are dropped so cold entries can lazy-load from SSD; memory-only nodes are
    removed once their tensors go, so node count stays bounded.
    """

    def __init__(
        self,
        capacity: int | dict[str, int] = 8,
        disk_dir: str | os.PathLike | None = None,
        chunk: int = 512,
        log=None,
    ):
        self.chunk = int(chunk)
        if self.chunk <= 0:
            raise ValueError(f"prefix-cache chunk must be positive, got {chunk!r}")

        if isinstance(capacity, dict):
            self.capacity = {str(k): int(v) for k, v in capacity.items()}
            self._default_capacity = max(self.capacity.values(), default=0)
        else:
            self.capacity = {"default": int(capacity)}
            self._default_capacity = int(capacity)

        self._root = Node(key=ROOT_KEY, pos=0)
        self._blocks: dict[str, Node] = {ROOT_KEY: self._root}
        self._clock = 0
        self._log = log
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        self._disk = (
            AsyncPrefixDiskStore(self.disk_dir, log=log) if self.disk_dir else None
        )
        self._load_disk()

    def _debug(self, msg: str) -> None:
        if self._log is not None:
            self._log(f"PREFIX DISK {msg}")

    # ---------------------------------------------------------------- keys

    def _iter_chain_keys(self, token_ids) -> Iterator[str]:
        """Yield chain keys for complete chunks of ``token_ids``."""
        parent = ROOT_KEY
        chunk = self.chunk
        for start in range(0, len(token_ids) - chunk + 1, chunk):
            parent = block_key(token_ids[start:start + chunk], parent=parent)
            yield parent

    def chain_keys(self, token_ids) -> list[str]:
        """Chain keys for every chunk boundary of ``token_ids``, root excluded."""
        return list(self._iter_chain_keys(token_ids))

    def key_at(self, token_ids, pos: int) -> str | None:
        """Chain key for the block ending at ``pos``, or None if unreachable."""
        if pos == 0:
            return ROOT_KEY
        if pos % self.chunk or pos > len(token_ids):
            return None
        parent = ROOT_KEY
        for start in range(0, pos, self.chunk):
            parent = block_key(token_ids[start:start + self.chunk], parent=parent)
        return parent

    # ---------------------------------------------------------------- disk load

    def _load_disk(self) -> None:
        """Restore record metadata only; tensor blobs stay lazy until a hit."""
        if self.disk_dir is None:
            return
        manifest = self.disk_dir / "manifest.json"
        legacy = self.disk_dir / "prefixcache.pkl"
        if manifest.exists():
            self._debug(f"LOAD SKIP incompatible_format path={manifest}")
        elif legacy.exists():
            self._debug(f"LOAD SKIP incompatible_legacy path={legacy}")
        if self._disk is None:
            return

        loaded = skipped = 0
        for record in sorted(self._disk.records(), key=lambda r: r.pos):
            if record.pos - record.start != self.chunk:
                skipped += 1
                continue
            parent = self._blocks.get(record.parent or ROOT_KEY)
            if parent is None:
                self._debug(f"LOAD SKIP missing_parent key={record.key[:12]}")
                continue
            node = self._node_from_record(record, parent)
            self._blocks[node.key] = node
            self._clock = max(self._clock, record.touch)
            loaded += 1
        if skipped:
            self._debug(f"LOAD SKIP chunk_mismatch records={skipped} "
                        f"chunk={self.chunk}")
        if loaded:
            self._evict()
            self._debug(f"LOAD records={loaded} path={self.disk_dir}")

    def _node_from_record(self, record: DiskBlockRecord, parent: Node) -> Node:
        return Node(
            key=record.key,
            pos=record.pos,
            parent=parent,
            source=record.source,
            pool=record.pool,
            touch=record.touch,
            reusable=record.ssm_spec is not None,
        )

    # ---------------------------------------------------------------- accounting

    def _capacity_for(self, pool: str) -> int:
        return self.capacity.get(pool, self._default_capacity)

    def _resident_entries(self) -> Iterator[Node]:
        for node in self._blocks.values():
            if node is not self._root and node.ssm is not None:
                yield node

    @staticmethod
    def _chain(node: Node) -> Iterator[Node]:
        cur: Node | None = node
        while cur is not None and cur.key != ROOT_KEY:
            yield cur
            cur = cur.parent

    def _resident_count(self, pool: str) -> int:
        return sum(1 for n in self._resident_entries() if n.pool == pool)

    def _evict(self) -> None:
        pools = set(self.capacity)
        pools.update(node.pool for node in self._resident_entries())
        for pool in pools:
            while self._resident_count(pool) > self._capacity_for(pool):
                victim = min(
                    (n for n in self._resident_entries() if n.pool == pool),
                    key=lambda n: n.touch,
                    default=None,
                )
                if victim is None:
                    break
                self._drop_payload(victim)
        self._drop_cold_blocks()

    def _drop_payload(self, node: Node) -> None:
        node.ssm = None
        node.cached_h = None

    def _drop_cold_blocks(self) -> None:
        """Release attention off every resident chain; forget unbacked nodes."""
        keep: set[str] = set()
        for node in self._resident_entries():
            for anc in self._chain(node):
                keep.add(anc.key)

        for key, node in list(self._blocks.items()):
            if node is self._root or key in keep:
                continue
            node.attn = None
            if self._disk_record(node) is None:
                del self._blocks[key]

    def prune_unreferenced(self) -> None:
        """Release cold payloads without touching resident chains."""
        self._drop_cold_blocks()

    # ---------------------------------------------------------------- lookup

    def find(self, token_ids) -> Match | None:
        """Return the deepest reusable prefix of ``token_ids``."""
        candidates: list[Node] = []
        for key in self._iter_chain_keys(token_ids):
            node = self._blocks.get(key)
            if node is None:
                break
            if node.ssm is not None or node.reusable:
                candidates.append(node)

        for best in reversed(candidates):
            if not self._load_reusable_payload(best):
                self._debug(f"MISS payload_unavailable pos={best.pos} "
                            f"key={best.key[:12]}")
                continue
            blocks = self._path_blocks(best)
            if blocks is None:
                self._debug(f"MISS attention_unavailable pos={best.pos} "
                            f"key={best.key[:12]}")
                continue

            self._clock += 1
            best.touch = self._clock
            payload = ("blocks", blocks, best.ssm, best.pos)
            if best.cached_h is not None:
                payload += (best.cached_h,)
            self._evict()
            return Match(prefix_len=best.pos, payload=payload, source=best.source,
                         pool=best.pool, key=best.key)
        return None

    def _path_blocks(self, node: Node) -> list[Any] | None:
        blocks = []
        for cur in self._chain(node):
            if cur.parent is None or not self._load_attention(cur):
                return None
            blocks.append(cur.attn)
        blocks.reverse()
        return blocks

    def _disk_record(self, node: Node) -> DiskBlockRecord | None:
        if self._disk is None:
            return None
        return self._disk.get_record(node.key)

    def _load_attention(self, node: Node) -> bool:
        if node.attn is not None:
            return True
        record = self._disk_record(node)
        if record is None:
            return False
        node.attn = self._disk.load_attn(record)
        return node.attn is not None

    def _load_reusable_payload(self, node: Node) -> bool:
        if node.ssm is None:
            record = self._disk_record(node)
            if record is None or record.ssm_spec is None:
                return False
            node.ssm = self._disk.load_ssm(record)
            node.reusable = True
            if node.cached_h is None and record.cached_h_spec is not None:
                node.cached_h = self._disk.load_cached_h(record)
        return node.ssm is not None

    # ---------------------------------------------------------------- store

    def store_block(
        self,
        full_prefix,
        start: int,
        pos: int,
        attn,
        *,
        ssm=None,
        source: str | None = None,
        pool: str = "default",
        cached_h=None,
        parent_key: str | None = None,
    ) -> str | None:
        """Add one chunk-aligned attention block; optionally make its end reusable."""
        start = int(start)
        pos = int(pos)
        if pos <= start or start < 0 or pos > len(full_prefix):
            raise ValueError("invalid prefix-cache block range")
        if pos - start != self.chunk or start % self.chunk:
            raise ValueError(
                f"prefix-cache block [{start}:{pos}] is not aligned to "
                f"chunk={self.chunk}"
            )

        if start == 0:
            parent_key = ROOT_KEY
        elif parent_key is None:
            parent_key = self.key_at(full_prefix, start)
        parent = self._blocks.get(parent_key) if parent_key is not None else None
        if parent is None or parent.pos != start:
            self._debug(f"STORE BLOCK SKIP missing_parent start={start} pos={pos}")
            return None

        key = block_key(full_prefix[start:pos], parent=parent_key)
        node = self._blocks.get(key)
        new_node = node is None
        if node is None:
            node = Node(key=key, pos=pos, parent=parent)
            self._blocks[key] = node
        node.parent = parent
        node.attn = attn

        if ssm is not None:
            self._clock += 1
            node.ssm = ssm
            node.source = source
            node.pool = str(pool)
            node.cached_h = cached_h
            node.reusable = True
            node.touch = self._clock
            self._evict()
        if self._disk is not None and (new_node or ssm is not None):
            self._disk.submit_block(
                key=key,
                parent=parent_key or None,
                tokens=full_prefix[start:pos],
                start=start,
                pos=pos,
                attn=attn,
                ssm=ssm,
                cached_h=cached_h,
                pool=node.pool,
                source=node.source,
                touch=node.touch,
            )
        return key

    def flush(self) -> None:
        if self._disk is not None:
            self._disk.flush()

    def close(self) -> None:
        if self._disk is not None:
            self._disk.close()
