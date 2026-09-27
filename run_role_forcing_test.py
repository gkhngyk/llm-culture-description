"""Runner for the Alternative A test (strengthened role-taking).

Commands:
  count    — print job count + cost estimate, no API calls
  run      — execute the 150 responder calls (resumable)
  analyze  — summarize + plot (no API calls)

Usage:
  python run_role_forcing_test.py count
  python run_role_forcing_test.py run --concurrency 8
  python run_role_forcing_test.py analyze

Does NOT touch ultimatum_sim.py or any existing results file.
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

from analysis.role_forcing_test import (
    make_jobs, pair_key, run_role_forced_pair,
    load_rows, summarize, interpret, plot_role_forcing,
)
from analysis.stats import load_jsonl

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS = PROJECT / "plots"
OUT_PATH = RESULTS / "role_forcing_test.jsonl"
ANALYSIS_PATH = RESULTS / "role_forcing_analysis.json"
PLOT_PATH = PLOTS / "12_role_forcing.png"


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


def _append(path: Path, result) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def cmd_count() -> None:
    jobs = make_jobs()
    print(f"[count] total jobs: {len(jobs)}  (responder-only)")
    # Rough cost estimate at Haiku pricing
    avg_in = 650  # slightly longer system prompt than baseline
    avg_out = 200
    est = len(jobs) * (avg_in * 1e-6 + avg_out * 5e-6)
    print(f"[count] estimated cost: ~${est:.2f}")
    print(f"[count] condition: role_forced (system prompt only)")


def cmd_run(concurrency: int) -> None:
    jobs = make_jobs()
    done = _completed_keys(OUT_PATH)
    remaining = [
        j for j in jobs
        if pair_key(j["arm"], j["offer"], j["pair_idx"]) not in done
    ]
    print(f"[run] total={len(jobs)} done={len(done)} remaining={len(remaining)}")
    if not remaining:
        print("[run] nothing to do")
        return

    t0 = time.time()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(run_role_forced_pair, **j): j for j in remaining}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                res = fut.result()
                _append(OUT_PATH, res)
                completed += 1
                if completed % 15 == 0 or completed == len(remaining):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (len(remaining) - completed) / max(rate, 1e-6)
                    print(f"[run] {completed}/{len(remaining)} "
                          f"rate={rate:.1f}/s eta={eta:.0f}s failed={failed}")
            except Exception as e:
                failed += 1
                pid = pair_key(j["arm"], j["offer"], j["pair_idx"])
                print(f"[run] FAIL {pid}: {e}", file=sys.stderr)
    print(f"[run] finished completed={completed} failed={failed} "
          f"elapsed={time.time()-t0:.1f}s")


def cmd_analyze() -> None:
    if not OUT_PATH.exists():
        print(f"[analyze] no results at {OUT_PATH}", file=sys.stderr)
        sys.exit(1)
    rows = load_rows(OUT_PATH)
    primary = load_jsonl(RESULTS / "primary.jsonl")
    summary = summarize(rows, primary)

    print(f"[analyze] forced n={len(rows)}")
    print("[analyze] cell-level comparison:")
    print(f"  {'arm':<12} {'offer':>6} {'baseline':>10} {'forced':>10} {'delta':>10}")
    for row in summary["table"]:
        print(f"  {row['arm']:<12} %{row['offer']:>4} "
              f"{row['baseline_rate']:>10.4f} {row['forced_rate']:>10.4f} "
              f"{row['delta']:>+10.4f}")

    print("\n[analyze] arm-level pooling:")
    for arm, stats in summary["by_arm"].items():
        print(f"  {arm:<12} baseline={stats['baseline_rate']:.4f} "
              f"forced={stats['forced_rate']:.4f} "
              f"delta={stats['delta']:+.4f}")

    pooled = summary["pooled"]
    print(f"\n[analyze] POOLED:")
    print(f"  baseline rate: {pooled['baseline_rate']:.4f} "
          f"({pooled['baseline_rejected']}/{pooled['baseline_n']})")
    print(f"  forced rate:   {pooled['forced_rate']:.4f} "
          f"({pooled['forced_rejected']}/{pooled['forced_n']})")
    print(f"  delta:         {pooled['delta']:+.4f}")

    interpretation = interpret(pooled)
    print(f"\n[analyze] INTERPRETATION:\n  {interpretation}")

    out = {
        "n_forced": len(rows),
        "summary": summary,
        "interpretation": interpretation,
    }
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[analyze] wrote {ANALYSIS_PATH}")

    plot_role_forcing(summary, PLOT_PATH)
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
