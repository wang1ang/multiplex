#!/usr/bin/env python3
"""Small fixed-model benchmark for prefill and generation speed.

The benchmark intentionally uses the local Qwen model used during optimization
work and two representative Chinese prompts.  It reports only prefill and
generation throughput. One short warmup run per prompt is performed by default
so first-use MLX/Metal JIT compilation is not charged to the measured generation
loop.

Run from this directory:
    .venv/bin/python benchmark_inference.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from multiplex.kernel.engine import Engine
from multiplex.kernel.mtp import find_drafter
from multiplex.kernel.scheduler import (
    DEFAULT_PREFILL_CHUNK,
    PrefillGroup,
    Req,
    Scheduler,
)

MODEL = "Qwen3.6-27B-Q4-MTPLX-v2-Q2Mix11-L29UpQ4-Q3KO16"
MODEL_PATH = Path.home() / ".mtplx" / "models" / MODEL
DATASET_PATH = Path(__file__).with_name("benchmarks") / "prompts.json"


def prompt_ids(tokenizer, text: str, target_length: int = 512) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Keep the benchmark input length fixed, as in the challenge. Repeating
    # the deterministic domain seed avoids model/tokenizer-specific token IDs.
    if len(ids) < target_length:
        ids = (ids * ((target_length + len(ids) - 1) // len(ids)))[:target_length]
    return ids[:target_length]


def run_once(engine: Engine, text: str, max_tokens: int, *, k: int) -> dict[str, float | int]:
    ids = prompt_ids(engine.tokenizer, text)
    drafter = find_drafter(engine)
    req = Req(rid=0, prompt=ids, max_tokens=max_tokens)
    scheduler = Scheduler(
        engine,
        drafter,
        # Fixed decode window, matching the challenge timing contract: do not
        # stop the timed leg early on EOS.
        eos_token_ids=set(),
        k=k,
        chunk=512,
        debug=False,
        dynamic_depth=True,
    )
    group = PrefillGroup(req=req)

    t0 = time.perf_counter()
    while True:
        done = scheduler.prefill_chunk(group)
        if done is None:
            raise RuntimeError("prefill unexpectedly cancelled")
        if done:
            break
    scheduler.merge_ready(group)
    prefill_seconds = time.perf_counter() - t0

    # The first token is emitted by merge_ready(); decode timing starts after it.
    generated = 1
    t1 = time.perf_counter()
    while scheduler.has_rows():
        emitted = scheduler.step()
        generated += sum(len(tokens) for _, tokens in emitted)
    decode_seconds = time.perf_counter() - t1
    window_seconds = time.perf_counter() - t0
    return {
        "prompt_tokens": len(ids),
        "generated_tokens": generated,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "window_seconds": window_seconds,
        "prefill_tok_s": len(ids) / max(prefill_seconds, 1e-9),
        "decode_tok_s": max(generated - 1, 0) / max(decode_seconds, 1e-9),
        "cost_stats": scheduler.cost_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--max-tokens", type=int, default=512,
        help="decode window (default: 512; use -n 128 only for a quick smoke test)",
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--limit", type=int, default=2,
                        help="only run the first N prompts (default: 2; 0 = all 8)")
    args = parser.parse_args()

    if not MODEL_PATH.is_dir():
        raise SystemExit(f"model not found: {MODEL_PATH}")
    print(f"[model] {MODEL_PATH}")
    engine = Engine(str(MODEL_PATH))
    print(f"[loaded] {engine.load_seconds:.1f}s")
    print(f"[mtp] {'yes' if find_drafter(engine) else 'no'}")

    # Warm only the prefill path; decode/MTP warmup is intentionally omitted.
    engine.warmup(prompt_length=DEFAULT_PREFILL_CHUNK)
    entries = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit > 0:
        entries = entries[:args.limit]
    speedups = []
    for entry in entries:
        name, text = entry["id"], entry["prompt"]
        serial = run_once(engine, text, args.max_tokens, k=0)
        candidate = run_once(engine, text, args.max_tokens, k=3)
        # Challenge-equivalent score: seed prefill is charged inside the
        # measured decode window; there is no hand-picked prefill weight.
        speedup = serial["window_seconds"] / max(candidate["window_seconds"], 1e-9)
        speedups.append(speedup)
        print(f"\n[prompt] {name}")
        print(f"prompt={candidate['prompt_tokens']} tok, generated={candidate['generated_tokens']} tok")
        print(f"serial window: {serial['window_seconds']:.3f}s (prefill {serial['prefill_tok_s']:.1f}, generation {serial['decode_tok_s']:.1f} tok/s)")
        print(f"MTP window: {candidate['window_seconds']:.3f}s (prefill {candidate['prefill_tok_s']:.1f}, generation {candidate['decode_tok_s']:.1f} tok/s)")
        print(f"raw speedup: {speedup:.3f}x")
        for depth, stat in sorted(candidate["cost_stats"].items()):
            print(
                f"cost D{depth}: {stat['seconds_per_round'] * 1000:.1f} ms/round, "
                f"{stat['tokens_per_second']:.1f} committed tok/s, "
                f"acceptance={stat['acceptance']:.2f}"
            )
    print(f"\n[median raw speedup] {statistics.median(speedups):.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
