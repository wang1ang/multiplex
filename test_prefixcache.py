"""Unit tests for the prefix-cache policy: chunk-chain keying + LRU.

Pure policy tests — payloads are plain MLX arrays, no model or engine involved.
Run: .venv/bin/python test_prefixcache.py
"""

from __future__ import annotations

import tempfile

import mlx.core as mx

from multiplex.kernel.prefixcache.disk import block_key
from multiplex.kernel.prefixcache.policy import PrefixCache


CHUNK = 4

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


# ---------------------------------------------------------------- chain keys

print("\nchain keys")
cache = PrefixCache(capacity=64, chunk=CHUNK)
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
cache = PrefixCache(capacity=64, chunk=CHUNK)
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

# ---------------------------------------------------------------- pool isolation

print("\npool isolation")
cache = PrefixCache(capacity={"prompt": 2, "session": 64}, chunk=CHUNK)
for i in range(6):
    cache.store_block(ids(4, base=i * 10), 0, 4, payload(1024),
                      ssm=payload(1024), pool="prompt")
for i in range(6):
    cache.store_block(ids(4, base=500 + i * 10), 0, 4, payload(1024),
                      ssm=payload(1024), pool="session")

n_prompt = sum(1 for n in cache._resident_entries() if n.pool == "prompt")
n_session = sum(1 for n in cache._resident_entries() if n.pool == "session")
check("tight pool evicted", n_prompt <= 2, n_prompt)
check("roomy pool untouched by other pool's churn", n_session == 6, n_session)

# ---------------------------------------------------------------- node hygiene

print("\nnode hygiene")
cache = PrefixCache(capacity=8, chunk=CHUNK)
for i in range(50):
    cache.store_block(ids(4, base=i * 10), 0, 4, payload(1024),
                      ssm=payload(1024), pool="prompt")
check("memory-only nodes do not accumulate", len(cache._blocks) < 20,
      len(cache._blocks))

# Blocks stored without an SSM are not reusable and get reclaimed on prune.
cache = PrefixCache(capacity=64, chunk=CHUNK)
cache.store_block(tokens, 0, 4, payload(1024))
cache.prune_unreferenced()
check("non-reusable block pruned", len(cache._blocks) == 1, len(cache._blocks))

# ---------------------------------------------------------------- alignment

print("\nalignment guards")
cache = PrefixCache(capacity=64, chunk=CHUNK)
for start, end, why in [(0, 3, "short block"), (1, 5, "unaligned start"),
                        (0, 8, "double-length block")]:
    try:
        cache.store_block(ids(8), start, end, payload(16))
        check(f"rejects {why}", False, "accepted")
    except ValueError:
        check(f"rejects {why}", True)

try:
    PrefixCache(capacity=64, chunk=0)
    check("rejects chunk=0", False)
except ValueError:
    check("rejects chunk=0", True)

# ---------------------------------------------------------------- disk round-trip

print("\ndisk round-trip")
with tempfile.TemporaryDirectory() as d:
    c1 = PrefixCache(capacity=64, disk_dir=d, chunk=CHUNK)
    store_chain(c1, tokens, 8, attn_bytes=64, ssm_bytes=64)
    c1.flush()
    c1.close()

    c2 = PrefixCache(capacity=64, disk_dir=d, chunk=CHUNK)
    m = c2.find(tokens)
    check("reloads from disk", m is not None and m.prefix_len == 8,
          m and m.prefix_len)
    check("lazy-loaded chain is complete", m is not None and len(m.payload[1]) == 2)
    restored = m.payload[1][0][0][0] if m else None
    check("tensor values survive", restored is not None
          and bool(mx.all(restored == 1.0).item()))
    c2.close()

    # A cache dir written with a different chunk must not be half-adopted.
    c3 = PrefixCache(capacity=64, disk_dir=d, chunk=CHUNK * 2)
    check("chunk mismatch ignored", c3.find(tokens) is None)
    c3.close()


print()
if failures:
    print(f"FAILED {len(failures)}: {', '.join(failures)}")
    raise SystemExit(1)
print("all prefix-cache policy tests passed")
