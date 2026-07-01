"""Engine smoke test: batch + next-k.

The engine only does correct batched forwards; the caller samples. We use greedy
(argmax) here as the caller's sampling.

Verifies:
  1. Batched decode == single-sequence decode, token-for-token (equal length).
  2. next-k: feeding k tokens at once returns k positions of logits, and each
     position's argmax equals feeding those tokens one at a time.

Run:  python smoke_engine.py [model_path]
"""

import sys
import mlx.core as mx
from slipstream.engine import Engine

DEFAULT_MODEL = "~/.mtplx/models/Agents-A1-MTPLX"
N = 12


def argmax_last(logits, row, pos):
    return int(mx.argmax(logits[row, pos]))


def greedy_decode(eng, prompt_ids, n):
    """Single-seq greedy via the engine (B=1), caller-side sampling."""
    st, logits = eng.prefill([prompt_ids])
    out = [argmax_last(logits, 0, len(prompt_ids) - 1)]
    for _ in range(n - 1):
        logits = eng.forward(st, mx.array([[out[-1]]]))
        out.append(argmax_last(logits, 0, 0))
    return out


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    eng = Engine(model_path)
    print(f"[load] {eng.load_seconds:.1f}s")

    pa = eng.encode("The capital of France is")
    pb = eng.encode("List some colors now okay")
    print(f"[len] a={len(pa)} b={len(pb)}")

    ref_a = greedy_decode(eng, pa, N)
    ref_b = greedy_decode(eng, pb, N)

    # (1) batched decode == solo
    st, logits = eng.prefill([pa, pb])
    outs = [[argmax_last(logits, i, len(p) - 1)] for i, p in enumerate([pa, pb])]
    for _ in range(N - 1):
        toks = mx.array([[outs[0][-1]], [outs[1][-1]]])
        logits = eng.forward(st, toks)
        outs[0].append(argmax_last(logits, 0, 0))
        outs[1].append(argmax_last(logits, 1, 0))
    m_a, m_b = outs[0] == ref_a, outs[1] == ref_b
    print(f"[1 batch==solo] A={m_a} B={m_b}")

    # reference chain: first token + next 3, fed one-by-one
    stR, lgR = eng.prefill([pa])
    ref = [argmax_last(lgR, 0, len(pa) - 1)]
    for _ in range(3):
        logits = eng.forward(stR, mx.array([[ref[-1]]]))
        ref.append(argmax_last(logits, 0, 0))

    # (2) next-k: feed ref[0:3] at once; position j predicts ref[j+1]
    st2, _ = eng.prefill([pa])
    lgk = eng.forward(st2, mx.array([[ref[0], ref[1], ref[2]]]))
    preds = [argmax_last(lgk, 0, j) for j in range(3)]
    m_k = lgk.shape[1] == 3 and preds == ref[1:4]
    print(f"[2 next-k] k={lgk.shape[1]} preds match one-by-one: {m_k}")

    ok = m_a and m_b and m_k
    print("[engine]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
