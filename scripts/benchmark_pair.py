#!/usr/bin/env python3
"""Thermally fair, sequential MTPLX speed comparison.

Each model is loaded in a separate MTPLX process (never simultaneously), warmed
and measured ``runs`` times, then the process exits before the cooldown.  The
reported values are medians, so model load time is excluded by MTPLX's tok/s
measurements.

Example:
  python scripts/benchmark_pair.py BASE CANDIDATE --runs 3 --cooldown 60
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, tempfile, time
from pathlib import Path


def one(mtplex: str, model: str, suite: str, max_tokens: int, root: Path, run: int) -> dict:
    out = root / f"{Path(model).name}-{run}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [mtplex, "forge", "verify", model, "--out", str(out),
           "--run-id", f"bench-{run}", "--max-tokens", str(max_tokens),
           "--suite", suite, "--json"]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"MTPLX failed for {model}:\n{p.stderr}\n{p.stdout}")
    # forge --json emits one JSON object; tolerate informational lines.
    match = re.search(r"(\{\s*\"rows\"[\s\S]*\})\s*$", p.stdout)
    if not match:
        raise RuntimeError(f"Could not parse MTPLX JSON for {model}: {p.stdout[-1000:]}")
    rows = json.loads(match.group(1))["rows"]
    d0 = next(r for r in rows if r["depth"] == 0)
    d3 = next((r for r in rows if r["depth"] == 3), rows[-1])
    return {"model": model, "run": run, "ar_tok_s": d0["tok_s"],
            "d3_tok_s": d3["tok_s"], "d3_multiplier": d3.get("multiplier_vs_ar"),
            "acceptance": d3.get("acceptance_by_position", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--mtplx", default=os.environ.get("MTPLX_BIN", "mtplx"))
    ap.add_argument("--suite", default="MTPLX/mtplx/benchmarks/prompts/calibration_coding.jsonl")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--output", type=Path, default=Path("benchmark-pair.json"))
    a = ap.parse_args()
    models = [a.baseline, a.candidate]
    rows = []
    with tempfile.TemporaryDirectory(prefix="mtplx-pair-") as td:
        root = Path(td)
        for i in range(a.runs):
            for model in models if i % 2 == 0 else reversed(models):
                rows.append(one(a.mtplx, model, a.suite, a.max_tokens, root, len(rows)))
                if a.cooldown:
                    time.sleep(a.cooldown)
    summary = {"baseline": a.baseline, "candidate": a.candidate,
               "runs": a.runs, "cooldown_s": a.cooldown, "rows": rows}
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    for model in models:
        vals = [r["d3_tok_s"] for r in rows if r["model"] == model]
        vals.sort(); med = vals[len(vals)//2]
        print(f"{Path(model).name}: D3 median {med:.2f} tok/s (runs={len(vals)})")
    print(f"saved {a.output}")

if __name__ == "__main__":
    main()
