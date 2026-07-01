"""Interactive REPL to feel the batched engine. Loads the model once.

    python try_engine.py [--model PATH] [--raw] [--temp T] [-n N]

Enter prompts one per line; a BLANK line runs them together as a batch. One
prompt streams live; multiple prompts decode as a batch and print together.
Commands (any time):
    :n <int>      set max tokens
    :temp <float> set temperature (0 = greedy)
    :raw          toggle chat-template on/off
    :seed <int>   set sampling seed
    :q            quit

Sampling is done here (the engine only does forwards).
"""

import argparse
import sys
import time

import mlx.core as mx
from slipstream.engine import Engine

MODEL = "~/.mtplx/models/Agents-A1-MTPLX"


def sample(row, temp):
    if temp <= 0:
        return int(mx.argmax(row))
    return int(mx.random.categorical(mx.log(mx.softmax(row * (1.0 / temp)))))


def to_ids(eng, text, raw):
    if raw:
        return eng.encode(text)
    return eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )


def run_batch(eng, prompts, cfg):
    """Generate for a batch of prompts. Streams live when there's a single one."""
    prompt_ids = [to_ids(eng, p, cfg["raw"]) for p in prompts]
    lens = [len(p) for p in prompt_ids]
    if len(set(lens)) > 1:
        print(f"[warn] unequal lengths {lens}; padded", file=sys.stderr)

    B = len(prompt_ids)
    eos = eng.eos_token_ids
    done = [False] * B
    produced = [[] for _ in range(B)]

    def take(logits, positions):
        """Sample one token per active row from logits[i, positions[i]]."""
        for i in range(B):
            if done[i]:
                continue
            t = sample(logits[i, positions[i]], cfg["temp"])
            produced[i].append(t)
            if t in eos:
                done[i] = True

    # All rows decode together (one forward/step). Display streams ONE row at a
    # time, in order: paint the current row's new text; when it's finished, move
    # to the next row (its background-generated text flushes, then streams on).
    cur = 0
    shown = ""

    def paint():
        nonlocal cur, shown
        while cur < B:
            full = eng.decode(produced[cur])
            if full != shown:
                print(full[len(shown):], end="", flush=True)
                shown = full
            if not (done[cur] or len(produced[cur]) >= cfg["n"]):
                return
            cur += 1
            shown = ""
            if cur < B:
                print(f"\n\n--- prompt {cur + 1}: {prompts[cur][:50]!r}")

    if B > 1:
        print(f"--- prompt 1: {prompts[0][:50]!r}")
    t0 = time.time()
    state, logits = eng.prefill(prompt_ids)
    take(logits, [n - 1 for n in lens])
    paint()
    while cur < B:
        logits = eng.forward(state, mx.array([[p[-1]] for p in produced]))
        take(logits, [0] * B)
        paint()

    print()
    dt = time.time() - t0
    total = sum(map(len, produced))
    print(f"[{total} tok, {dt:.1f}s, {total/dt:.1f} tok/s]", file=sys.stderr)


def _emit(eng, tokens, already):
    """Print the newly-decoded suffix of `tokens`, return updated full text."""
    full = eng.decode(tokens)
    if full != already:
        print(full[len(already):], end="", flush=True)
    return full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("-n", "--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    eng = Engine(args.model)
    print(f"[loaded in {eng.load_seconds:.1f}s]")
    print("Enter prompts (one per line). Blank line runs the batch. "
          ":q quit, :help commands.")

    cfg = {"n": args.max_tokens, "temp": args.temp, "raw": args.raw}
    buf = []  # collected prompts for the next batch

    while True:
        try:
            # index the next slot so you can see how many are queued
            line = input(f"{len(buf)+1}> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # blank line -> run whatever is collected (1 prompt streams; >1 batches)
        if line.strip() == "":
            if buf:
                run_batch(eng, buf, cfg)
                buf = []
            continue

        s = line.strip()
        if s in (":q", ":quit", ":exit"):
            break
        if s in (":help", ":h"):
            print(__doc__)
            continue
        if s == ":raw":
            cfg["raw"] = not cfg["raw"]
            print(f"[raw={cfg['raw']}]")
            continue
        if s.startswith(":n "):
            cfg["n"] = int(s[3:]); continue
        if s.startswith(":temp "):
            cfg["temp"] = float(s[6:]); continue
        if s.startswith(":seed "):
            mx.random.seed(int(s[6:])); continue
        if s.startswith(":"):
            print(f"[unknown command {s!r}; :help for list]")
            continue

        buf.append(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
