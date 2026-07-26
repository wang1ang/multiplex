"""End-to-end prefix-cache correctness against a real model.

The invariant: reusing a cached prefix must produce the SAME tokens as prefilling
that prompt cold. A restore bug (wrong chain, off-by-one position, mismatched
SSM/MTP state) shows up as divergent output, not as an error.

Runs with k=0 (pure AR) for exact token equality, then k>0 to exercise the MTP
block path. Also checks reuse across a process restart via the disk store.

Run:  .venv/bin/python test/test_prefixcache_e2e.py [model_path]
"""

import os
import shutil
import sys
import tempfile

from multiplex.engine import Engine
from multiplex.mtp import find_drafter
from multiplex.scheduler import Scheduler, Req, PrefillGroup

# TODO: this default is one developer's local checkout, so the test cannot run
# anywhere else without an argument. Make model_path required, or skip cleanly
# when it is absent, before this lands anywhere others run tests.
MODEL = os.path.expanduser("~/.mtplx/models/Agents-A1-MTPLX")

# Prefill stores blocks on chunk boundaries only, and the prompt pool needs
# >4096 tokens (PROMPT_CACHE_MIN_TOKENS), so the shared prefix must be long.
CHUNK = 512

failures = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        failures.append(name)


def run(sch, prompt_ids, n):
    """Run one request to completion through a scheduler; return its tokens."""
    g = PrefillGroup(req=Req(0, prompt_ids, n))
    while not sch.prefill_chunk(g):
        pass
    sch.merge_ready(g)
    out = list(g.req.out)
    while sch.has_rows():
        for _rid, toks in sch.step():
            out.extend(toks)
    return out


def make_sched(eng, dr, *, k, budget, disk=None, chunk=CHUNK):
    return Scheduler(eng, dr, eos_token_ids=eng.tokenizer.eos_token_ids, k=k,
                     chunk=chunk, prefix_cache=budget, prefix_cache_dir=disk)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL
    eng = Engine(model_path)
    dr = find_drafter(eng)
    print(f"[model={os.path.basename(model_path)} "
          f"drafter={'yes' if dr else 'none (AR)'}]")

    tok = eng.tokenizer
    # The prompt pool only caches prompts over PROMPT_CACHE_MIN_TOKENS (4096),
    # and only on chunk boundaries, so the shared prefix must clear 8 chunks for
    # any block to become reusable. 12 chunks leaves headroom above the floor.
    filler = ("The quick brown fox jumps over the lazy dog. " * 700)
    base = tok.encode(filler)
    if len(base) < 12 * CHUNK:
        raise SystemExit(f"filler too short: {len(base)} < {12 * CHUNK} tokens")
    shared = base[:12 * CHUNK]
    tail_a = tok.encode(" In summary, the answer is")
    tail_b = tok.encode(" Therefore we conclude that")
    print(f"[shared prefix={len(shared)} tok]")

    N = 24

    for k in (0, 3) if dr else (0,):
        label = f"k={k}"
        print(f"\nreuse correctness ({label})")

        # Cold reference: fresh scheduler, caching disabled entirely.
        ref_a = run(make_sched(eng, dr, k=k, budget=0), shared + tail_a, N)
        ref_b = run(make_sched(eng, dr, k=k, budget=0), shared + tail_b, N)

        # Warm: one scheduler, prompt A populates the cache, then A and B reuse it.
        sch = make_sched(eng, dr, k=k, budget="2GiB")
        got_a1 = run(sch, shared + tail_a, N)
        entries = len(list(sch.prefix_cache.cache._resident_entries()))
        check(f"{label} blocks stored", entries > 0, f"entries={entries}")


        # Assert the hit BEFORE rerunning: if nothing is reused, the
        # "matches cold" checks below pass vacuously.
        reused_a = sch.prefix_cache.cache.find(shared + tail_a)
        reused_b = sch.prefix_cache.cache.find(shared + tail_b)
        check(f"{label} A reuses a prefix",
              reused_a is not None and reused_a.prefix_len >= 8 * CHUNK,
              f"prefix_len={reused_a and reused_a.prefix_len}")
        check(f"{label} B reuses A's prefix",
              reused_b is not None and reused_b.prefix_len >= 8 * CHUNK,
              f"prefix_len={reused_b and reused_b.prefix_len}")

        got_a2 = run(sch, shared + tail_a, N)
        got_b = run(sch, shared + tail_b, N)

        check(f"{label} cold run matches reference", got_a1 == ref_a,
              f"\n    got={got_a1[:8]}\n    ref={ref_a[:8]}")
        check(f"{label} exact-prefix reuse matches cold", got_a2 == ref_a,
              f"\n    got={got_a2[:8]}\n    ref={ref_a[:8]}")
        check(f"{label} divergent-tail reuse matches cold", got_b == ref_b,
              f"\n    got={got_b[:8]}\n    ref={ref_b[:8]}")

        # Byte budget must actually bound residency under churn, and a cache
        # squeezed to its floor must stay correct (fall back to cold prefill).
        tiny = make_sched(eng, dr, k=k, budget=1024**2)
        got_tiny = run(tiny, shared + tail_a, N)
        check(f"{label} correct under 1MiB budget", got_tiny == ref_a,
              f"\n    got={got_tiny[:8]}\n    ref={ref_a[:8]}")
        resident = tiny.prefix_cache.cache.resident_bytes("prompt")
        check(f"{label} 1MiB budget respected", resident <= 1024**2,
              f"resident={resident}")

    print("\ndisk reuse across restart")
    d = tempfile.mkdtemp(prefix="mpx-e2e-")
    try:
        ref = run(make_sched(eng, dr, k=0, budget=0), shared + tail_a, N)

        warm = make_sched(eng, dr, k=0, budget="2GiB", disk=d)
        run(warm, shared + tail_a, N)
        warm.prefix_cache.cache.flush()
        warm.prefix_cache.cache.close()

        # Fresh cache object over the same dir: metadata only, tensors lazy.
        cold = make_sched(eng, dr, k=0, budget="2GiB", disk=d)
        m = cold.prefix_cache.cache.find(shared + tail_a)
        check("reloaded prefix found", m is not None and m.prefix_len >= CHUNK,
              f"prefix_len={m and m.prefix_len}")
        got = run(cold, shared + tail_a, N)
        check("disk-restored reuse matches cold", got == ref,
              f"\n    got={got[:8]}\n    ref={ref[:8]}")
        cold.prefix_cache.cache.close()

        # A dir written at chunk=512 must not be adopted at a different chunk.
        other = make_sched(eng, dr, k=0, budget="2GiB", disk=d, chunk=256)
        got_other = run(other, shared + tail_a, N)
        check("chunk change stays correct", got_other == ref,
              f"\n    got={got_other[:8]}\n    ref={ref[:8]}")
        other.prefix_cache.cache.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED {len(failures)}: {', '.join(failures)}")
        return 1
    print("all prefix-cache e2e tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
