"""Unit tests for the prefix-cache policy: chain keying + byte-budget LRU.

Pure policy tests — payloads are plain MLX arrays, no model or engine involved.
Run: .venv/bin/python test_prefixcache.py
"""

from __future__ import annotations

import tempfile

import mlx.core as mx

from multiplex.kernel.prefixcache.disk import block_key, spec_nbytes, encode_tree
from multiplex.kernel.prefixcache.policy import (
    MIN_BYTE_BUDGET,
    PrefixCache,
    parse_bytes,
)


CHUNK = 4
GiB = 1024**3

failures: list[str] = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def payload(nbytes: int, *, fill: float = 1.0):
    """One attention-block-shaped payload of a known size.

    Evaluated here because ``store_block`` requires it: the disk writer thread
    cannot drive MLX's thread-bound GPU stream.
    """
    n = max(1, nbytes // 4)
    arr = mx.full((n,), fill, dtype=mx.float32)
    mx.eval(arr)
    return [[arr]]


def ids(n: int, *, base: int = 0):
    return [base + i for i in range(n)]


def store_chain(cache, tokens, upto, *, pool="prompt", ssm_bytes=4, attn_bytes=4):
    """Store every chunk block of tokens[:upto]; last block gets the SSM payload."""
    for start in range(0, upto, cache.chunk):
        end = start + cache.chunk
        last = end == upto
        cache.store_block(
            tokens, start, end, payload(attn_bytes),
            ssm=payload(ssm_bytes) if last else None,
            source=f"{pool}@{end}" if last else None,
            pool=pool,
        )


# ---------------------------------------------------------------- byte parsing

print("parse_bytes")
check("plain int", parse_bytes(4 * GiB) == 4 * GiB)
check("GiB suffix", parse_bytes("4GiB") == 4 * GiB, parse_bytes("4GiB"))
check("MiB suffix", parse_bytes("512MiB") == 512 * 1024**2)
check("GB is decimal", parse_bytes("1GB") == 1000**3)
check("fractional", parse_bytes("1.5GiB") == int(1.5 * GiB))
check("case/space insensitive", parse_bytes(" 2 gib ") == 2 * GiB)
check("zero disables", parse_bytes(0) == 0)

for bad, why in [(8, "old entry count"), ("8", "old count as str"),
                 (-1, "negative"), ("4XiB", "bad unit"), ("", "empty"),
                 (True, "bool")]:
    try:
        parse_bytes(bad)
        check(f"rejects {why}", False, f"accepted {bad!r}")
    except (ValueError, TypeError):
        check(f"rejects {why}", True)

check("floor is 1MiB", parse_bytes(MIN_BYTE_BUDGET) == MIN_BYTE_BUDGET)

# ---------------------------------------------------------------- chain keys

print("\nchain keys")
cache = PrefixCache(budget="1GiB", chunk=CHUNK)
tokens = ids(12)
keys = cache.chain_keys(tokens)
check("one key per full chunk", len(keys) == 3, len(keys))
check("keys are distinct", len(set(keys)) == 3)

# Same chunk under a different parent must not collide — this is the property
# that makes a fixed-size key safe to substitute for the whole token prefix.
alt = ids(4, base=100) + ids(4, base=4)
check(
    "same chunk, different parent -> different key",
    cache.chain_keys(alt)[1] != keys[1],
)
check("prefix stability", cache.chain_keys(ids(8)) == keys[:2])
check(
    "chain extends parent",
    block_key(tokens[4:8], parent=keys[0]) == keys[1],
)
check("partial tail chunk ignored", len(cache.chain_keys(ids(11))) == 2)

# ---------------------------------------------------------------- match depth

print("\nlongest-prefix match")
cache = PrefixCache(budget="1GiB", chunk=CHUNK)
store_chain(cache, tokens, 4)
store_chain(cache, tokens, 8)
store_chain(cache, tokens, 12)

m = cache.find(tokens)
check("exact request hits deepest", m is not None and m.prefix_len == 12,
      m and m.prefix_len)
check("payload carries whole chain", m is not None and len(m.payload[1]) == 3,
      m and len(m.payload[1]))

m = cache.find(ids(8) + ids(4, base=900))
check("divergent tail falls back to 8", m is not None and m.prefix_len == 8,
      m and m.prefix_len)
check("shallower payload is shorter", m is not None and len(m.payload[1]) == 2)

check("total miss returns None", cache.find(ids(12, base=500)) is None)
m = cache.find(ids(2))
check("shorter than one chunk misses", m is None)

# A request longer than anything stored still reuses the deepest stored block.
m = cache.find(tokens + ids(8, base=12))
check("longer request reuses stored depth", m is not None and m.prefix_len == 12,
      m and m.prefix_len)

# ---------------------------------------------------------------- byte budget

print("\nbyte-budget eviction")
# Budget fits ~2 of these entries: each is 1MiB of SSM plus its chain attention.
MB = 1024**2
cache = PrefixCache(budget=3 * MB, chunk=CHUNK)
for i in range(4):
    toks = ids(4, base=1000 * (i + 1))
    cache.store_block(toks, 0, 4, payload(1024), ssm=payload(MB),
                      source=f"e{i}", pool="prompt")

resident = list(cache._resident_entries())
check("evicts by bytes, not count", len(resident) <= 2, len(resident))
check("stays within budget", cache.resident_bytes("prompt") <= 3 * MB,
      cache.resident_bytes("prompt"))
check("keeps the newest", any(n.source == "e3" for n in resident),
      [n.source for n in resident])
check("dropped the oldest", not any(n.source == "e0" for n in resident))

# One huge entry costs as much as many small ones: the old count-based budget
# treated these identically, which is the bug this change fixes.
small = PrefixCache(budget=100 * MB, chunk=CHUNK)
for i in range(8):
    small.store_block(ids(4, base=i * 10), 0, 4, payload(1024),
                      ssm=payload(MB), pool="prompt")
check("8 small entries all fit", len(list(small._resident_entries())) == 8,
      len(list(small._resident_entries())))

big = PrefixCache(budget=100 * MB, chunk=CHUNK)
for i in range(8):
    big.store_block(ids(4, base=i * 10), 0, 4, payload(1024),
                    ssm=payload(30 * MB), pool="prompt")
check("8 huge entries do not", len(list(big._resident_entries())) < 8,
      len(list(big._resident_entries())))

# ---------------------------------------------------------------- pool isolation

print("\npool isolation")
cache = PrefixCache(budget={"prompt": 2 * MB, "session": 64 * MB}, chunk=CHUNK)
for i in range(6):
    cache.store_block(ids(4, base=i * 10), 0, 4, payload(1024),
                      ssm=payload(MB), pool="prompt")
for i in range(6):
    cache.store_block(ids(4, base=500 + i * 10), 0, 4, payload(1024),
                      ssm=payload(MB), pool="session")

n_prompt = sum(1 for n in cache._resident_entries() if n.pool == "prompt")
n_session = sum(1 for n in cache._resident_entries() if n.pool == "session")
check("tight pool evicted", n_prompt <= 2, n_prompt)
check("roomy pool untouched by other pool's churn", n_session == 6, n_session)

# ---------------------------------------------------------------- chain sharing

print("\nchain accounting")
cache = PrefixCache(budget="1GiB", chunk=CHUNK)
store_chain(cache, tokens, 12, attn_bytes=1024, ssm_bytes=1024)
# One entry at pos=12 over a 3-block chain: its payload plus 3 attention blocks.
check("single entry prices its whole chain",
      cache.resident_bytes() == 1024 + 3 * 1024,
      cache.resident_bytes())

# Three nested entries share that same chain: attention must be counted once
# overall, not once per entry, or nesting would look 3x more expensive.
cache = PrefixCache(budget="1GiB", chunk=CHUNK)
for upto in (4, 8, 12):
    store_chain(cache, tokens, upto, attn_bytes=1024, ssm_bytes=1024)
check("three entries", len(list(cache._resident_entries())) == 3,
      len(list(cache._resident_entries())))
check("shared attention counted once",
      cache.resident_bytes() == 3 * 1024 + 3 * 1024,
      cache.resident_bytes())

# ---------------------------------------------------------------- node hygiene

print("\nnode hygiene")
cache = PrefixCache(budget=2 * MB, chunk=CHUNK)
for i in range(50):
    cache.store_block(ids(4, base=i * 10), 0, 4, payload(1024),
                      ssm=payload(MB), pool="prompt")
check("memory-only nodes do not accumulate", len(cache._blocks) < 20,
      len(cache._blocks))

# Blocks stored without an SSM are not reusable and get reclaimed on prune.
cache = PrefixCache(budget="1GiB", chunk=CHUNK)
cache.store_block(tokens, 0, 4, payload(1024))
cache.prune_unreferenced()
check("non-reusable block pruned", len(cache._blocks) == 1, len(cache._blocks))

# ---------------------------------------------------------------- alignment

print("\nalignment guards")
cache = PrefixCache(budget="1GiB", chunk=CHUNK)
for start, end, why in [(0, 3, "short block"), (1, 5, "unaligned start"),
                        (0, 8, "double-length block")]:
    try:
        cache.store_block(ids(8), start, end, payload(16))
        check(f"rejects {why}", False, "accepted")
    except ValueError:
        check(f"rejects {why}", True)

try:
    PrefixCache(budget="1GiB", chunk=0)
    check("rejects chunk=0", False)
except ValueError:
    check("rejects chunk=0", True)

# ---------------------------------------------------------------- disk round-trip

print("\ndisk round-trip")
with tempfile.TemporaryDirectory() as d:
    c1 = PrefixCache(budget="1GiB", disk_dir=d, chunk=CHUNK)
    store_chain(c1, tokens, 8, attn_bytes=64, ssm_bytes=64)
    c1.flush()
    c1.close()

    c2 = PrefixCache(budget="1GiB", disk_dir=d, chunk=CHUNK)
    m = c2.find(tokens)
    check("reloads from disk", m is not None and m.prefix_len == 8,
          m and m.prefix_len)
    check("lazy-loaded chain is complete", m is not None and len(m.payload[1]) == 2)
    restored = m.payload[1][0][0][0] if m else None
    check("tensor values survive", restored is not None
          and bool(mx.all(restored == 1.0).item()))
    c2.close()

    # A cache dir written with a different chunk must not be half-adopted.
    c3 = PrefixCache(budget="1GiB", disk_dir=d, chunk=CHUNK * 2)
    check("chunk mismatch ignored", c3.find(tokens) is None)
    c3.close()

print("\nspec_nbytes")
spec, blobs = encode_tree(payload(4096))
check("spec bytes match tensor bytes", spec_nbytes(spec) == 4096,
      spec_nbytes(spec))
check("blocked spec bytes match",
      spec_nbytes(encode_tree(
          [[mx.zeros((1, 1, 2048, 4), dtype=mx.float32)]], block_size=256)[0]
      ) == 1 * 1 * 2048 * 4 * 4)

print()
if failures:
    print(f"FAILED {len(failures)}: {', '.join(failures)}")
    raise SystemExit(1)
print("all prefix-cache policy tests passed")
