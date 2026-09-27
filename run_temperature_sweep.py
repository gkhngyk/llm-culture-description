"""Runner for the temperature robustness sweep (follow-up analysis #1).

Commands:
  count     — print total job count & cost estimate, no API calls
  run       — execute the full sweep (resumable via pair_id)
  analyze   — summarize + plot (no API calls)

Usage:
  python run_temperature_sweep.py count
  python run_temperature_sweep.py run --concurrency 8
  python run_temperature_sweep.py analyze

Does NOT touch results/primary.jsonl or any existing sweep files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultimatum_sim import run_pair, PairResult
from analysis.temperature_sweep import (
    make_temperature_jobs, pair_key, load_sweep, summarize,
    pooled_by_temperature, plot_temperature_sweep,
)
from analysis.stats import load_jsonl

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS = PROJECT / "plots"
OUT_PATH = RESULTS / "temperature_sweep.jsonl"
ANALYSIS_PATH = RESULTS / "temperature_sweep_analysis.json"
PLOT_PATH = PLOTS / "10_temperature_sweep.png"


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["pair_id"])
            except Exception:
                pass
    return keys


def _append(path: Path, result: PairResult) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _run_one(job: dict) -> PairResult:
    """Run one pair, rewrite pair_id to include temperature tag."""
    # Ensure pair_idx doesn't collide with primary format
    res = run_pair(**job)
    # Rewrite pair_id so it doesn't clash with primary
    new_id = pair_key(job)
    res.pair_id = new_id
    return res


def cmd_count() -> None:
    jobs = make_temperature_jobs()
    print(f"[count] total jobs: {len(jobs)}")
    print(f"[count] role calls: {len(jobs)}  (responder-only)")
    # Rough Haiku-scale cost estimate (other models similar magnitude)
    avg_in_tokens = 600
    avg_out_tokens = 200
    # Conservative blended: ~$1.5/M in + $6/M out averaged across 3 models
    est_usd = len(jobs) * (avg_in_tokens * 1.5e-6 + avg_out_tokens * 6e-6)
    print(f"[count] estimated cost: ~${est_usd:.2f}")
    by_t: dict = {}
    for j in jobs:
        by_t[j["temperature"]] = by_t.get(j["temperature"], 0) + 1
    for t, n in sorted(by_t.items()):
        print(f"  T={t}: {n} jobs")


def cmd_run(concurrency: int) -> None:
    jobs = make_temperature_jobs()
    done = _completed_keys(OUT_PATH)
    remaining = [j for j in jobs if pair_key(j) not in done]
    print(f"[run] total={len(jobs)} done={len(done)} remaining={len(remaining)}")
    if not remaining:
        print("[run] nothing to do")
        return

    t0 = time.time()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_run_one, j): j for j in remaining}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                res = fut.result()
                _append(OUT_PATH, res)
                completed += 1
                if completed % 25 == 0 or completed == len(remaining):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (len(remaining) - completed) / max(rate, 1e-6)
                    print(f"[run] {completed}/{len(remaining)} "
                          f"rate={rate:.1f}/s eta={eta:.0f}s failed={failed}")
            except Exception as e:
                failed += 1
                print(f"[run] FAIL {pair_key(j)}: {e}", file=sys.stderr)
    print(f"[run] finished completed={completed} failed={failed} "
          f"elapsed={time.time()-t0:.1f}s")


def cmd_analyze() -> None:
    if not OUT_PATH.exists():
        print(f"[analyze] no results at {OUT_PATH}", file=sys.stderr)
        sys.exit(1)
    rows = load_sweep(OUT_PATH)
    primary = load_jsonl(RESULTS / "primary.jsonl")
    cells = summarize(rows)
    pooled = pooled_by_temperature(cells)

    out = {
        "n_total": len(rows),
        "cells": cells,
        "pooled_by_temperature": pooled,
    }
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[analyze] n={len(rows)}")
    print("[analyze] pooled rejection rates by temperature × arm:")
    for k, v in sorted(pooled.items()):
        print(f"  {k:20s}  n={v['n']:3d}  rejected={v['rejected']:2d}  "
              f"rate={v['rate']:.4f}")

    plot_temperature_sweep(rows, primary, PLOT_PATH)
    print(f"[analyze] plot: {PLOT_PATH}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["count", "run", "analyze"])
    p.add_argument("--concurrency", type=int, default=8)
    a = p.parse_args()
    {
        "count":   lambda: cmd_count(),
        "run":     lambda: cmd_run(a.concurrency),
        "analyze": lambda: cmd_analyze(),
    }[a.command]()


if __name__ == "__main__":
    main()
