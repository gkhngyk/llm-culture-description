"""
Jev repeatability: is P(accept) stable for an identical request?

Jev has no temperature knob, so before the full grid we measure how much the
returned noul moves when the exact same state + question is sent repeatedly,
and compare that to the spread between different profiles in the same cell.

Design: 2 arms x 5 offers x 3 stored responder profiles (from the haiku rows
of results/primary.jsonl) x 10 identical repeats = 300 calls.

Outputs:
  results/jev_repeatability.jsonl
  results/jev_repeatability_analysis.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from jev_client import jev_responder, profile_from_row

PROJECT_DIR = Path(__file__).parent
RESULTS = PROJECT_DIR / "results"
OUT = RESULTS / "jev_repeatability.jsonl"
ANALYSIS = RESULTS / "jev_repeatability_analysis.json"

N_PROFILES = 3
N_REPEATS = 10
SOURCE_MODEL = "anthropic/claude-haiku-4.5"


def _select_rows() -> list[dict]:
    rows = [json.loads(l) for l in open(RESULTS / "primary.jsonl")]
    rows = [r for r in rows if r["model_slug"] == SOURCE_MODEL]
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["arm"], r["offer"])].append(r)
    picked = []
    for cell in sorted(by_cell):
        picked += sorted(by_cell[cell], key=lambda r: r["pair_id"])[:N_PROFILES]
    return picked


def _job(row: dict, rep: int) -> dict:
    sid = f"gabm-ultimatum-{row['arm']}-off{row['offer']}-jev-rep-{row['pair_id']}-r{rep}"
    out = jev_responder(profile_from_row(row), row["offer"], session_id=sid,
                        profile_variant=row["profile_variant"],
                        prompt_variant=row["prompt_variant"])
    out.update({"source_pair_id": row["pair_id"], "repeat": rep})
    return out


def run() -> list[dict]:
    rows = _select_rows()
    jobs = [(r, k) for r in rows for k in range(N_REPEATS)]
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex, open(OUT, "w") as f:
        futs = [ex.submit(_job, r, k) for r, k in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            f.write(json.dumps(res) + "\n")
            results.append(res)
    return results


def analyze(results: list[dict]) -> dict:
    by_profile = defaultdict(list)
    for r in results:
        by_profile[(r["arm"], r["offer"], r["source_pair_id"])].append(r["p_accept"])

    within_sd = []        # SD across identical repeats, per profile
    distinct_values = []  # how many distinct p values across repeats
    profile_means = defaultdict(list)
    per_profile = []
    for (arm, offer, pid), ps in sorted(by_profile.items()):
        ps = np.array(ps)
        within_sd.append(ps.std(ddof=1))
        distinct_values.append(len(set(ps.round(6))))
        profile_means[(arm, offer)].append(ps.mean())
        per_profile.append({"arm": arm, "offer": offer, "pair_id": pid,
                            "mean": round(float(ps.mean()), 4),
                            "sd": round(float(ps.std(ddof=1)), 4),
                            "min": float(ps.min()), "max": float(ps.max()),
                            "n_distinct": int(len(set(ps.round(6))))})

    between_sd = [np.std(m, ddof=1) for m in profile_means.values()]

    # Variance decomposition over all calls: repeat noise vs profile vs cell.
    all_p = np.array([r["p_accept"] for r in results])
    noise_var = float(np.mean(np.square(within_sd)))
    total_var = float(all_p.var(ddof=1))

    return {
        "n_calls": len(results),
        "n_profiles": len(by_profile),
        "repeats_per_profile": N_REPEATS,
        "model_snapshots": sorted({r["model_snapshot"] for r in results}),
        "mean_within_profile_sd": round(float(np.mean(within_sd)), 5),
        "max_within_profile_sd": round(float(np.max(within_sd)), 5),
        "profiles_fully_deterministic": int(sum(d == 1 for d in distinct_values)),
        "mean_between_profile_sd_within_cell": round(float(np.mean(between_sd)), 5),
        "repeat_noise_share_of_total_variance": round(noise_var / total_var, 5) if total_var else None,
        "total_cost_usd": round(sum((r["usage"] or {}).get("cost", 0) for r in results), 6),
        "per_profile": per_profile,
    }


if __name__ == "__main__":
    res = run()
    summary = analyze(res)
    ANALYSIS.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_profile"}, indent=2))
    for p in summary["per_profile"]:
        print(f"{p['arm']:>10} off{p['offer']:<3} {p['pair_id'][-4:]}  "
              f"mean={p['mean']:.3f} sd={p['sd']:.4f} range=[{p['min']:.2f},{p['max']:.2f}] "
              f"distinct={p['n_distinct']}")
