"""Prefix-cache policy: block-tree longest-prefix matching + per-pool LRU.

This module has no model knowledge. It only tracks token prefixes and opaque
payloads supplied by the L3 state adapter.

The core invariant is:
  * attention KV is stored as per-block deltas on tree edges;
  * SSM is stored on reusable boundary nodes;
  * a match restores the parent-chain attention blocks plus that node's SSM.

Disk persistence is intentionally detached for now. ``disk_dir`` is accepted so
call sites keep their shape, but old on-disk formats are only reported and then
ignored until the new tree format is designed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


@dataclass
class Node:
    pos: int
    attn: Any | None = None
    ssm: Any | None = None
    start: int = 0
    prefix: tuple[int, ...] = ()
    parent: "Node | None" = None
    children: dict[tuple[int, ...], "Node"] = field(default_factory=dict)
    source: str | None = None
    pool: str = "default"
    cached_h: Any | None = None
    touch: int = 0


@dataclass
class Match:
    prefix_len: int
    payload: Any
    source: str | None = None
    pool: str = "default"


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    entries: list[Node] = field(default_factory=list)


class PrefixCache:
    """Block-tree prefix cache with independent LRU pools.

    ``capacity`` bounds reusable boundary nodes per pool. Attention-only ancestor
    blocks are retained only while a reusable descendant needs them.
    """

    def __init__(
        self,
        capacity: int | dict[str, int] = 8,
        disk_dir: str | os.PathLike | None = None,
        log=None,
    ):
        if isinstance(capacity, dict):
            self.capacity = {str(k): int(v) for k, v in capacity.items()}
            self._default_capacity = max(self.capacity.values(), default=0)
        else:
            self.capacity = {"default": int(capacity)}
            self._default_capacity = int(capacity)

        self._block_root = Node(pos=0, prefix=())
        self._blocks: dict[tuple[int, ...], Node] = {(): self._block_root}
        self._root = _TrieNode()
        self._clock = 0
        self._log = log
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        self._load_disk()

    def _debug(self, msg: str) -> None:
        if self._log is not None:
            self._log(f"PREFIX DISK {msg}")

    def _load_disk(self) -> None:
        """Skip incompatible disk formats until tree persistence is redesigned."""
        if self.disk_dir is None:
            return
        manifest = self.disk_dir / "manifest.json"
        legacy = self.disk_dir / "prefixcache.pkl"
        if manifest.exists():
            self._debug(f"LOAD SKIP incompatible_format path={manifest}")
        elif legacy.exists():
            self._debug(f"LOAD SKIP incompatible_legacy path={legacy}")

    def _capacity_for(self, pool: str) -> int:
        return self.capacity.get(pool, self._default_capacity)

    def iter_entries(self):
        for node in self._blocks.values():
            if node is not self._block_root and node.ssm is not None:
                yield node

    def _entry_count(self, pool: str | None = None) -> int:
        if pool is None:
            return sum(1 for _ in self.iter_entries())
        return sum(1 for node in self.iter_entries() if node.pool == pool)

    def _rebuild_index(self) -> None:
        self._root = _TrieNode()
        for node in self.iter_entries():
            self._index_block_entry(node)

    def _index_block_entry(self, node: Node) -> None:
        cur = self._root
        for tok in node.prefix:
            cur = cur.children.setdefault(tok, _TrieNode())
        cur.entries.append(node)

    def _evict(self) -> None:
        changed = False
        pools = set(self.capacity)
        pools.update(node.pool for node in self.iter_entries())
        for pool in pools:
            while self._entry_count(pool) > self._capacity_for(pool):
                victim = min(
                    (node for node in self.iter_entries() if node.pool == pool),
                    key=lambda node: node.touch,
                    default=None,
                )
                if victim is None:
                    break
                victim.ssm = None
                victim.cached_h = None
                victim.source = None
                self._prune_block(victim)
                changed = True
        if changed:
            self._rebuild_index()

    def _prune_block(self, node: Node) -> None:
        """Drop leaf attention blocks that no reusable descendant needs."""
        while node is not self._block_root and not node.children and node.ssm is None:
            parent = node.parent
            if parent is None:
                return
            parent.children.pop(tuple(node.prefix[node.start:node.pos]), None)
            self._blocks.pop(node.prefix, None)
            node = parent

    def prune_unreferenced(self) -> None:
        """Drop attention-only leaves that are not anchoring a reusable node."""
        changed = False
        for node in list(self._blocks.values()):
            if node is self._block_root:
                continue
            if node.ssm is None and not node.children:
                self._prune_block(node)
                changed = True
        if changed:
            self._rebuild_index()

    def find(self, token_ids) -> Match | None:
        """Return the deepest reusable prefix of ``token_ids``."""
        token_ids = tuple(token_ids)
        best: Node | None = None
        cur = self._root
        for tok in token_ids:
            cur = cur.children.get(tok)
            if cur is None:
                break
            if cur.entries:
                candidate = max(cur.entries, key=lambda node: node.touch)
                if (
                    best is None
                    or candidate.pos > best.pos
                    or (candidate.pos == best.pos and candidate.touch > best.touch)
                ):
                    best = candidate
        if best is None:
            return None

        self._clock += 1
        best.touch = self._clock
        blocks = self._path_blocks(best)
        if blocks is None:
            best.ssm = None
            best.cached_h = None
            best.source = None
            self._rebuild_index()
            return self.find(token_ids)

        payload = ("blocks", blocks, best.ssm, best.pos)
        if best.cached_h is not None:
            payload += (best.cached_h,)
        return Match(prefix_len=best.pos, payload=payload, source=best.source,
                     pool=best.pool)

    def _path_blocks(self, node: Node) -> list[Any] | None:
        blocks = []
        cur = node
        while cur is not self._block_root:
            if cur.attn is None or cur.parent is None:
                return None
            blocks.append(cur.attn)
            cur = cur.parent
        blocks.reverse()
        return blocks

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
    ) -> bool:
        """Add one attention block and optionally make its end reusable."""
        full_prefix = tuple(full_prefix)
        start = int(start)
        pos = int(pos)
        if pos <= start or start < 0 or pos > len(full_prefix):
            raise ValueError("invalid prefix-cache block range")

        parent_prefix = full_prefix[:start]
        prefix = full_prefix[:pos]
        parent = self._blocks.get(parent_prefix)
        if parent is None:
            self._debug(f"STORE BLOCK SKIP missing_parent start={start} pos={pos}")
            return False

        key = tuple(full_prefix[start:pos])
        node = self._blocks.get(prefix)
        if node is None:
            node = Node(pos=pos, start=start, prefix=prefix, parent=parent, attn=attn)
            self._blocks[prefix] = node
        else:
            node.pos = pos
            node.start = start
            node.prefix = prefix
            node.parent = parent
            node.attn = attn
        parent.children[key] = node

        if ssm is not None:
            self._clock += 1
            node.ssm = ssm
            node.source = source
            node.pool = str(pool)
            node.cached_h = cached_h
            node.touch = self._clock
            self._evict()
            self._rebuild_index()
        return True
