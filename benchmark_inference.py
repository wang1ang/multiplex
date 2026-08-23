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
import time
from pathlib import Path

from multiplex.kernel.engine import Engine
from multiplex.kernel.mtp import find_drafter
from multiplex.kernel.scheduler import PrefillGroup, Req, Scheduler

MODEL = "Qwen3.6-27B-Q4-MTPLX-v2-Q2Mix11-L29UpQ4-Q3KO16"
MODEL_PATH = Path.home() / ".mtplx" / "models" / MODEL
PROMPTS = (
    "你是什么模型？",
    "用python帮我写一个最简单的二叉树前序遍历",
)


def prompt_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_once(engine: Engine, text: str, max_tokens: int, *,
             prefill_last_logits: bool = True) -> dict[str, float | int]:
    ids = prompt_ids(engine.tokenizer, text)
    drafter = find_drafter(engine)
    req = Req(rid=0, prompt=ids, max_tokens=max_tokens)
    scheduler = Scheduler(
        engine,
        drafter,
        eos_token_ids=engine.tokenizer.eos_token_ids,
        k=3,
        chunk=512,
        debug=False,
        dynamic_depth=True,
        prefill_last_logits=prefill_last_logits,
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
    return {
        "prompt_tokens": len(ids),
        "generated_tokens": generated,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tok_s": len(ids) / max(prefill_seconds, 1e-9),
        "decode_tok_s": max(generated - 1, 0) / max(decode_seconds, 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-tokens", type=int, default=64)
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="do not run one short unmeasured generation per prompt",
    )
    args = parser.parse_args()

    if not MODEL_PATH.is_dir():
        raise SystemExit(f"model not found: {MODEL_PATH}")
    print(f"[model] {MODEL_PATH}")
    engine = Engine(str(MODEL_PATH))
    print(f"[loaded] {engine.load_seconds:.1f}s")
    print(f"[mtp] {'yes' if find_drafter(engine) else 'no'}")

    for text in PROMPTS:
        if not args.no_warmup:
            run_once(engine, text, min(8, args.max_tokens), prefill_last_logits=False)
            run_once(engine, text, min(8, args.max_tokens), prefill_last_logits=True)
        legacy = run_once(engine, text, args.max_tokens, prefill_last_logits=False)
        result = run_once(engine, text, args.max_tokens, prefill_last_logits=True)
        speedup = legacy["prefill_seconds"] / max(result["prefill_seconds"], 1e-9)
        print(f"\n[prompt] {text}")
        print("prompt={prompt_tokens} tok, generated={generated_tokens} tok".format(**result))
        print("prefill (legacy):  {prefill_tok_s:.1f} tok/s ({prefill_seconds:.3f}s)".format(**legacy))
        print("prefill (optimized): {prefill_tok_s:.1f} tok/s ({prefill_seconds:.3f}s)".format(**result))
        print(f"prefill speedup: {speedup:.2f}x")
        print("generation: {decode_tok_s:.1f} tok/s ({decode_seconds:.3f}s)".format(**result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
