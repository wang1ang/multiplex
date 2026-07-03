"""Engine smoke test: batch + next-k.

The engine only does correct batched forwards (returning hidden); the caller
applies lm_head + samples. We use greedy (argmax) as the caller's sampling.

Verifies:
  1. Batched decode == single-sequence decode, token-for-token (equal length).
  2. next-k: feeding k tokens at once returns k positions, and each position's
     argmax equals feeding those tokens one at a time.

Run:  python smoke_engine.py [model_path]
"""

import os
import sys
import mlx.core as mx
from multiplex.engine import Engine

DEFAULT_MODEL = os.path.expanduser("~/.mtplx/models/Agents-A1-MTPLX")
N = 12


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    eng = Engine(model_path)
    print(f"[load] {eng.load_seconds:.1f}s")

    def tok_at(hidden, row, pos):
        return int(mx.argmax(eng.logits(hidden[row, pos])))

    def greedy(prompt_ids):
        st, h = eng.prefill([prompt_ids])
        out = [tok_at(h, 0, len(prompt_ids) - 1)]
        for _ in range(N - 1):
            h = eng.forward(st, mx.array([[out[-1]]]))
            out.append(tok_at(h, 0, 0))
        return out

    pa = eng.encode("The capital of France is")
    pb = eng.encode("List some colors now okay")
    print(f"[len] a={len(pa)} b={len(pb)}")
    ref_a, ref_b = greedy(pa), greedy(pb)

    # (1) batched decode == solo
    st, h = eng.prefill([pa, pb])
    outs = [[tok_at(h, i, len(p) - 1)] for i, p in enumerate([pa, pb])]
    for _ in range(N - 1):
        h = eng.forward(st, mx.array([[outs[0][-1]], [outs[1][-1]]]))
        outs[0].append(tok_at(h, 0, 0))
        outs[1].append(tok_at(h, 1, 0))
    m_a, m_b = outs[0] == ref_a, outs[1] == ref_b
    print(f"[1 batch==solo] A={m_a} B={m_b}")

    # reference chain: first token + next 3, one-by-one
    stR, h = eng.prefill([pa])
    ref = [tok_at(h, 0, len(pa) - 1)]
    for _ in range(3):
        h = eng.forward(stR, mx.array([[ref[-1]]]))
        ref.append(tok_at(h, 0, 0))

    # (2) next-k: feed ref[0:3] at once; position j predicts ref[j+1]
    st2, _ = eng.prefill([pa])
    hk = eng.forward(st2, mx.array([[ref[0], ref[1], ref[2]]]))
    preds = [tok_at(hk, 0, j) for j in range(3)]
    m_k = hk.shape[1] == 3 and preds == ref[1:4]
    print(f"[2 next-k] k={hk.shape[1]} preds match one-by-one: {m_k}")

    ok = m_a and m_b and m_k
    print("[engine]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
