"""Standalone DFlash speculative-decoding smoke test.

Runs the vendored DFlash MLX drafter (multiplex.kernel.dflash) against a local
target model, end-to-end: intermediate-hidden capture, block draft, verify,
accept, and (for GatedDelta targets) state rollback. Single-sequence path,
matching the official reference. Batched-scheduler integration is separate.

    python try_dflash.py --target <dir> --draft <dir> [--prompt ...] [--max-tokens N]
"""

import argparse
import time

from multiplex.kernel.dflash import load, load_draft, stream_generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target model dir or HF id")
    ap.add_argument("--draft", required=True, help="DFlash draft dir or HF id")
    ap.add_argument("--prompt", default="Write a quicksort in Python.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--block-size", type=int, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tokenizer = load(args.target)
    draft = load_draft(args.draft)
    print(f"[load] target+draft in {time.perf_counter() - t0:.1f}s")

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    accepted_lens, n_blocks = [], 0
    final = None
    print("-" * 60)
    for resp in stream_generate(
        model, draft, tokenizer, prompt,
        block_size=args.block_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ):
        print(resp.text, end="", flush=True)
        final = resp
        if resp.accepted:
            accepted_lens.append(int(resp.accepted))
            n_blocks += 1
    print("\n" + "-" * 60)
    if final is not None:
        mean_acc = sum(accepted_lens) / len(accepted_lens) if accepted_lens else 0.0
        print(f"[done] {final.generation_tokens} tok @ {final.generation_tps:.1f} tok/s | "
              f"blocks={n_blocks} mean_accept={mean_acc:.2f} | "
              f"peak={final.peak_memory:.1f}GB | finish={final.finish_reason}")


if __name__ == "__main__":
    main()
