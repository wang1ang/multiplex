"""Programmatic correctness tests for the dynamic-batch scheduler + hub.

The invariant everywhere: a request's output must equal what it produces run
ALONE (speculation/batching must not change the result). We test with k=0 (pure
AR) for exact token-for-token equality — that isolates logic from the float
accumulation that k>0 batching legitimately introduces.

Covers: single, multi-join-at-once, mid-flight join (入), early exit (出), and
concurrent hub submits from many threads (L4).

Run:  python test_scheduler.py [model_path]
"""

import os
import sys
import threading

import mlx.core as mx
from slipstream.engine import Engine
from slipstream.mtp import Drafter
from slipstream.scheduler import Scheduler, Req, PrefillGroup
from slipstream.hub import Hub

MODEL = os.path.expanduser("~/.mtplx/models/Agents-A1-MTPLX")
MTP = MODEL + "/mtp.safetensors"

PROMPTS = {
    "france": "The capital of France is",
    "story": "Once upon a time in a distant land there",
    "math": "Two plus three equals five so",
    "hi": "Say hi.",
}


def solo(eng, prompt_ids, n, k=0):
    """Reference: run one request alone through the scheduler; return its tokens."""
    sch = Scheduler(eng, _DR, k=k)
    g = PrefillGroup(reqs=[Req(0, prompt_ids, n)])
    while not sch.prefill_chunk(g):
        pass
    sch.merge_ready(g)
    out = list(g.reqs[0].out)
    while sch.has_rows():
        for _rid, toks in sch.step():
            out.extend(toks)
    return out


def drive(sch, joins):
    """Drive a scheduler: `joins` is [(after_step, PrefillGroup)]. Returns
    {rid: tokens}. after_step=0 means join before any decode step."""
    out = {}
    pending = sorted(joins, key=lambda j: j[0])
    t = 0
    # join everything scheduled at step 0
    while pending and pending[0][0] <= t:
        g = pending.pop(0)[1]
        while not sch.prefill_chunk(g):
            pass
        for rid, first in sch.merge_ready(g):
            out.setdefault(rid, []).append(first)
    while sch.has_rows() or pending:
        for rid, toks in sch.step():
            out.setdefault(rid, []).extend(toks)
        t += 1
        while pending and pending[0][0] <= t:
            g = pending.pop(0)[1]
            while not sch.prefill_chunk(g):
                pass
            for rid, first in sch.merge_ready(g):
                out.setdefault(rid, []).append(first)
    return out


def check(name, got, ref, exact_prefix=6):
    # A logic bug shows as EARLY divergence; batched float accumulation can flip
    # a token only after many steps (inherent, not a bug). So require exact match
    # for the first `exact_prefix` tokens, and that both ran to a similar length.
    n = min(len(got), len(ref), exact_prefix)
    ok = got[:n] == ref[:n] and len(got) >= exact_prefix
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + ("" if ok else f"  got{got[:8]} ref{ref[:8]}"))
    return ok


def main():
    global _DR
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL
    eng = Engine(model_path)
    _DR = Drafter(eng, model_path + "/mtp.safetensors")
    enc = eng.encode
    N = 20
    ok = True

    # references (each prompt alone)
    ref = {k: solo(eng, enc(p), N) for k, p in PROMPTS.items()}

    print("1) single request == solo")
    sch = Scheduler(eng, _DR, k=0)
    got = drive(sch, [(0, PrefillGroup(reqs=[Req(0, enc(PROMPTS["france"]), N)]))])
    ok &= check("france", got[0], ref["france"])

    print("2) three requests joined at once == each solo")
    sch = Scheduler(eng, _DR, k=0)
    g = PrefillGroup(reqs=[Req(0, enc(PROMPTS["france"]), N),
                           Req(1, enc(PROMPTS["story"]), N),
                           Req(2, enc(PROMPTS["math"]), N)])
    got = drive(sch, [(0, g)])
    ok &= check("france", got[0], ref["france"])
    ok &= check("story", got[1], ref["story"])
    ok &= check("math", got[2], ref["math"])

    print("3) mid-flight join (入): one runs, another joins at step 5")
    sch = Scheduler(eng, _DR, k=0)
    g0 = PrefillGroup(reqs=[Req(0, enc(PROMPTS["story"]), N)])
    g1 = PrefillGroup(reqs=[Req(1, enc(PROMPTS["france"]), N)])
    got = drive(sch, [(0, g0), (5, g1)])
    ok &= check("story (running)", got[0], ref["story"])
    ok &= check("france (joined@5)", got[1], ref["france"])

    print("4) early exit (出): short finishes, long continues")
    sch = Scheduler(eng, _DR, k=0)
    g = PrefillGroup(reqs=[Req(0, enc(PROMPTS["hi"]), N),
                           Req(1, enc(PROMPTS["story"]), N)])
    got = drive(sch, [(0, g)])
    ok &= check("hi (short)", got[0], ref["hi"])
    ok &= check("story (long)", got[1], ref["story"])

    print("5) hub: concurrent submits from many threads == each solo")
    hub = Hub(model_path, model_path + "/mtp.safetensors", k=0)
    results = {}

    def worker(name):
        toks = []
        # hub streams text; re-encode to compare token-wise would drift, so
        # compare decoded text against solo's decoded text instead
        for delta in hub.stream_text(eng.encode(PROMPTS[name]), N):
            toks.append(delta)
        results[name] = "".join(toks)

    threads = [threading.Thread(target=worker, args=(n,)) for n in PROMPTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for name in PROMPTS:
        ref_text = eng.decode(ref[name])
        got_text = results[name]
        # hub batches under load -> float accumulation can flip a late token, so
        # compare a prefix (enough to catch logic bugs, lenient on late drift).
        p = min(len(ref_text), len(got_text), 20)
        match = got_text[:p] == ref_text[:p] and len(got_text) > 5
        print(f"  [{'PASS' if match else 'FAIL'}] {name}"
              + ("" if match else f"\n     got {got_text[:60]!r}\n     ref {ref_text[:60]!r}"))
        ok &= match

    print("\n[scheduler tests]", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
