#!/usr/bin/env python3
"""Paired fixed-D3 vs dynamic-depth benchmark on one loaded MLX model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import mlx.core as mx

from multiplex.engine import Engine
from multiplex.mtp import find_drafter
from multiplex.scheduler import PrefillGroup, Req, Scheduler


def load_prompts(paths: list[Path], limit_per_suite: int) -> list[dict]:
    rows = []
    seen = set()
    for path in paths:
        added = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"missing string prompt in {path}: {row}")
            prompt_id = str(row.get("id") or f"{path.stem}-{added}")
            if prompt_id in seen:
                continue
            seen.add(prompt_id)
            rows.append({**row, "id": prompt_id, "suite": path.stem})
            added += 1
            if limit_per_suite > 0 and added >= limit_per_suite:
                break
    if not rows:
        raise ValueError("no prompts loaded")
    return rows


def exact_depth_rounds(trials: list[int], max_depth: int) -> dict[str, int]:
    padded = list(trials[:max_depth]) + [0] * max(0, max_depth - len(trials))
    return {
        f"D{depth}": padded[depth - 1] - (
            padded[depth] if depth < max_depth else 0
        )
        for depth in range(1, max_depth + 1)
    }


def run_case(
    engine: Engine,
    drafter,
    prompt: str,
    *,
    max_tokens: int,
    max_depth: int,
    dynamic: bool,
) -> dict:
    tokenizer = engine.tokenizer
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    request = Req(0, prompt_ids, max_tokens, temperature=None)
    group = PrefillGroup(req=request)
    scheduler = Scheduler(
        engine,
        drafter,
        eos_token_ids=tokenizer.eos_token_ids,
        k=max_depth,
        chunk=512,
        prefix_cache=0,
        prefix_cache_dir=None,
        debug=False,
        dynamic_depth=dynamic,
    )
    while not scheduler.prefill_chunk(group):
        pass
    scheduler.merge_ready(group)
    depth_history = []
    started = time.perf_counter()
    while scheduler.has_rows():
        depth_history.append(scheduler.k)
        scheduler.step()
    wall_seconds = time.perf_counter() - started

    trials = list(request.accept_trials_by_depth)
    acceptance = [
        request.accept_counts[index] / value if value else 0.0
        for index, value in enumerate(trials)
    ]
    token_bytes = ",".join(map(str, request.out)).encode()
    return {
        "generated_tokens": len(request.out),
        "decode_wall_seconds": wall_seconds,
        "decode_tok_s": max(len(request.out) - 1, 0) / max(wall_seconds, 1e-9),
        "scheduler_tok_s": (
            request.advance_tokens / max(request.advance_seconds, 1e-9)
        ),
        "accepted_by_depth": list(request.accept_counts),
        "trials_by_depth": trials,
        "acceptance_by_depth": acceptance,
        "depth_rounds": exact_depth_rounds(trials, max_depth),
        "depth_history": depth_history,
        "output_token_prefix": list(request.out[:32]),
        "output_token_sha256": hashlib.sha256(token_bytes).hexdigest(),
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(max(value, 1e-9)) for value in values) / len(values))


def summarize(rows: list[dict], max_depth: int) -> dict:
    pairs = {}
    for row in rows:
        pairs.setdefault((row["repeat"], row["prompt_id"]), {})[row["mode"]] = row

    ratios = []
    prompt_ratios: dict[str, list[float]] = {}
    hash_mismatches = []
    for (repeat, prompt_id), pair in pairs.items():
        if set(pair) != {"fixed_d3", "dynamic"}:
            continue
        ratio = pair["dynamic"]["scheduler_tok_s"] / max(
            pair["fixed_d3"]["scheduler_tok_s"], 1e-9
        )
        ratios.append(ratio)
        prompt_ratios.setdefault(prompt_id, []).append(ratio)
        if (
            pair["dynamic"]["output_token_sha256"]
            != pair["fixed_d3"]["output_token_sha256"]
        ):
            hash_mismatches.append({"repeat": repeat, "prompt_id": prompt_id})

    dynamic_rows = [row for row in rows if row["mode"] == "dynamic"]
    depth_rounds = {
        f"D{depth}": sum(row["depth_rounds"][f"D{depth}"] for row in dynamic_rows)
        for depth in range(1, max_depth + 1)
    }
    total_rounds = max(sum(depth_rounds.values()), 1)
    return {
        "paired_case_count": len(ratios),
        "dynamic_wins": sum(ratio > 1.0 for ratio in ratios),
        "geomean_speedup_dynamic_vs_fixed_d3": geomean(ratios) if ratios else 0.0,
        "median_speedup_dynamic_vs_fixed_d3": (
            statistics.median(ratios) if ratios else 0.0
        ),
        "per_prompt_geomean_speedup": {
            prompt_id: geomean(values)
            for prompt_id, values in sorted(prompt_ratios.items())
        },
        "dynamic_depth_rounds": depth_rounds,
        "dynamic_depth_share": {
            depth: count / total_rounds for depth, count in depth_rounds.items()
        },
        "output_hash_mismatches": hash_mismatches,
    }


def atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit-per-suite", type=int, default=0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.depth < 2:
        raise ValueError("dynamic comparison needs --depth >= 2")
    prompt_paths = [Path(path).expanduser().resolve() for path in args.prompts]
    prompts = load_prompts(prompt_paths, args.limit_per_suite)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    engine = Engine(str(Path(args.model).expanduser().resolve()))
    drafter = find_drafter(engine)
    if drafter is None:
        raise ValueError("model has no MTP sidecar")

    # Compile each fixed shape before timing paired cases.
    warm_prompt = prompts[0]["prompt"]
    for depth in range(1, args.depth + 1):
        print(f"[warmup] D{depth}", flush=True)
        run_case(
            engine,
            drafter,
            warm_prompt,
            max_tokens=24,
            max_depth=depth,
            dynamic=False,
        )
        gc.collect()
        mx.clear_cache()
    time.sleep(max(args.settle_seconds, 0.0))

    result = {
        "format": "multiplex-dynamic-depth-benchmark-v1",
        "model": str(Path(args.model).expanduser().resolve()),
        "prompt_suites": [str(path) for path in prompt_paths],
        "prompt_ids": [row["id"] for row in prompts],
        "depth": args.depth,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "dynamic_depth_policy": {
            "start_depth": args.depth,
            "min_depth": 1,
            "window": 16,
            "min_samples": 8,
            "up_threshold": 0.80,
            "down_threshold": 0.50,
            "retry_cooldown": 24,
        },
        "rows": [],
    }
    modes = [("fixed_d3", False), ("dynamic", True)]
    for repeat in range(args.repeats):
        for index, prompt_row in enumerate(prompts):
            ordered = modes if (repeat + index) % 2 == 0 else list(reversed(modes))
            for mode, dynamic in ordered:
                row_max_tokens = min(
                    args.max_tokens,
                    int(prompt_row.get("max_tokens") or args.max_tokens),
                )
                print(
                    f"[run] repeat={repeat + 1} prompt={prompt_row['id']} "
                    f"mode={mode}",
                    flush=True,
                )
                metrics = run_case(
                    engine,
                    drafter,
                    prompt_row["prompt"],
                    max_tokens=row_max_tokens,
                    max_depth=args.depth,
                    dynamic=dynamic,
                )
                row = {
                    "repeat": repeat + 1,
                    "prompt_id": prompt_row["id"],
                    "category": prompt_row.get("category"),
                    "suite": prompt_row["suite"],
                    "mode": mode,
                    "max_tokens": row_max_tokens,
                    **metrics,
                }
                result["rows"].append(row)
                result["summary"] = summarize(result["rows"], args.depth)
                atomic_write(output, result)
                print(
                    f"[result] {metrics['scheduler_tok_s']:.2f} tok/s "
                    f"accept={metrics['acceptance_by_depth']} "
                    f"depths={metrics['depth_rounds']}",
                    flush=True,
                )
                gc.collect()
                mx.clear_cache()
                time.sleep(max(args.settle_seconds, 0.0))

    result["summary"] = summarize(result["rows"], args.depth)
    atomic_write(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
