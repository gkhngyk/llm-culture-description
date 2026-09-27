"""
Jev arm over the full primary grid, row-paired with the LLM arms.

For every row of results/primary.jsonl (3 models x 2 arms x 5 offers x 50
pairs = 1500), Jev is asked P(accept) for the SAME stored responder profile and
the SAME briefing text that LLM saw. Repeatability test showed per-call noise
SD ~0.011 (results/jev_repeatability_analysis.json), comparable to the
between-profile spread, so each row is averaged over N_REPEATS calls.

Outputs:
  results/jev_primary.jsonl          one line per source row (Jev mean + LLM + sigmoid)
  results/jev_primary_analysis.json  cell table, H1 gap, agreement metrics
  plots/jev_vs_llm_vs_sigmoid.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.sigmoid_baseline import sigmoid_reject_probability
from jev_client import jev_responder, profile_from_row

PROJECT_DIR = Path(__file__).parent
RESULTS = PROJECT_DIR / "results"
PLOTS = PROJECT_DIR / "plots"
OUT = RESULTS / "jev_primary.jsonl"
ANALYSIS = RESULTS / "jev_primary_analysis.json"
PLOT = PLOTS / "jev_vs_llm_vs_sigmoid.png"

N_REPEATS = 3
OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]


def _job(row: dict) -> dict:
    ps = []
    snapshots = set()
    cost = 0.0
    for k in range(N_REPEATS):
        sid = f"gabm-ultimatum-{row['arm']}-off{row['offer']}-jev-grid-{row['pair_id']}-r{k}"
        r = jev_responder(profile_from_row(row), row["offer"], session_id=sid,
                          profile_variant=row["profile_variant"],
                          prompt_variant=row["prompt_variant"])
        ps.append(r["p_reject"])
        snapshots.add(r["model_snapshot"])
        cost += (r["usage"] or {}).get("cost", 0)
    fs = float(row["responder_profile"]["fairness_salience"])
    return {
        "source_pair_id": row["pair_id"],
        "source_model": row["model_slug"],
        "arm": row["arm"],
        "offer": row["offer"],
        "fairness_salience": fs,
        "jev_p_reject": float(np.mean(ps)),
        "jev_p_reject_repeats": ps,
        "jev_snapshots": sorted(snapshots),
        "llm_rejected": int(row["rejected"]),
        "sigmoid_p_reject": sigmoid_reject_probability(fs, row["offer"] / 100.0),
        "cost_usd": cost,
    }


def run() -> list[dict]:
    rows = [json.loads(l) for l in open(RESULTS / "primary.jsonl")]
    done = {}
    if OUT.exists():  # resume: skip rows already written
        for l in open(OUT):
            d = json.loads(l)
            done[(d["source_model"], d["source_pair_id"])] = d
    todo = [r for r in rows if (r["model_slug"], r["pair_id"]) not in done]
    print(f"{len(done)} rows cached, {len(todo)} to run ({len(todo) * N_REPEATS} calls)")
    with ThreadPoolExecutor(max_workers=8) as ex, open(OUT, "a") as f:
        futs = [ex.submit(_job, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            f.write(json.dumps(res) + "\n")
            f.flush()
            done[(res["source_model"], res["source_pair_id"])] = res
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    return list(done.values())


def analyze(res: list[dict]) -> dict:
    models = sorted({r["source_model"] for r in res})
    cell = defaultdict(list)
    for r in res:
        cell[(r["arm"], r["offer"])].append(r)

    table = []
    for arm in ARMS:
        for off in OFFERS:
            rs = cell[(arm, off)]
            row = {
                "arm": arm, "offer": off, "n": len(rs),
                "jev_mean_p_reject": round(float(np.mean([r["jev_p_reject"] for r in rs])), 4),
                "sigmoid_mean_p_reject": round(float(np.mean([r["sigmoid_p_reject"] for r in rs])), 4),
            }
            for m in models:
                mr = [r["llm_rejected"] for r in rs if r["source_model"] == m]
                row[f"llm_reject_rate::{m}"] = round(float(np.mean(mr)), 4) if mr else None
            table.append(row)

    def _cell(arm, off, key):
        return next(t[key] for t in table if t["arm"] == arm and t["offer"] == off)

    # H1-style gap at the low-offer cell (offer 10 == the <20% condition).
    gap = {
        "jev": round(_cell("weird", 10, "jev_mean_p_reject") - _cell("smallscale", 10, "jev_mean_p_reject"), 4),
        "sigmoid": round(_cell("weird", 10, "sigmoid_mean_p_reject") - _cell("smallscale", 10, "sigmoid_mean_p_reject"), 4),
    }
    for m in models:
        k = f"llm_reject_rate::{m}"
        gap[m] = round(_cell("weird", 10, k) - _cell("smallscale", 10, k), 4)

    # Offer-10 arm difference for Jev: Welch t-test on row means.
    w10 = [r["jev_p_reject"] for r in cell[("weird", 10)]]
    s10 = [r["jev_p_reject"] for r in cell[("smallscale", 10)]]
    t10 = stats.ttest_ind(w10, s10, equal_var=False)

    jev = np.array([r["jev_p_reject"] for r in res])
    sig = np.array([r["sigmoid_p_reject"] for r in res])
    llm = np.array([r["llm_rejected"] for r in res])
    fs = np.array([r["fairness_salience"] for r in res])

    # Does Jev use the numeric fairness attribute? Partial out offer by
    # correlating within each (arm, offer) cell, then averaging.
    within_fs_r = []
    for rs in cell.values():
        a = np.array([r["fairness_salience"] for r in rs])
        b = np.array([r["jev_p_reject"] for r in rs])
        if a.std() > 0 and b.std() > 0:
            within_fs_r.append(stats.spearmanr(a, b).statistic)

    return {
        "n_rows": len(res),
        "n_calls": len(res) * N_REPEATS,
        "jev_snapshots": sorted({s for r in res for s in r["jev_snapshots"]}),
        "total_cost_usd": round(sum(r["cost_usd"] for r in res), 5),
        "cell_table": table,
        "low_offer_gap_weird_minus_smallscale_off10": gap,
        "jev_off10_welch": {"t": round(float(t10.statistic), 3), "p": float(t10.pvalue),
                            "weird_mean": round(float(np.mean(w10)), 4),
                            "smallscale_mean": round(float(np.mean(s10)), 4)},
        "jev_vs_sigmoid": {
            "pearson_r": round(float(stats.pearsonr(jev, sig).statistic), 4),
            "mean_abs_diff": round(float(np.mean(np.abs(jev - sig))), 4),
            "point_agreement_at_0.5": round(float(np.mean((jev >= 0.5) == (sig >= 0.5))), 4),
        },
        "jev_vs_llm": {
            "point_agreement_at_0.5": round(float(np.mean((jev >= 0.5).astype(int) == llm)), 4),
            "jev_rows_above_0.5": int((jev >= 0.5).sum()),
            "llm_rejections": int(llm.sum()),
        },
        "jev_fairness_salience_spearman_within_cell_mean": round(float(np.mean(within_fs_r)), 4),
        "jev_fairness_salience_spearman_pooled": round(float(stats.spearmanr(fs, jev).statistic), 4),
    }


def plot(summary: dict) -> None:
    PLOTS.mkdir(exist_ok=True)
    table = summary["cell_table"]
    models = [k.split("::")[1] for k in table[0] if k.startswith("llm_reject_rate::")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, arm in zip(axes, ARMS):
        t = [r for r in table if r["arm"] == arm]
        x = [r["offer"] for r in t]
        ax.plot(x, [r["jev_mean_p_reject"] for r in t], "o-", lw=2.2, label="Jev 1.13 (mean P(reject))")
        ax.plot(x, [r["sigmoid_mean_p_reject"] for r in t], "s--", lw=1.6, color="gray",
                label="attribute sigmoid baseline")
        for m in models:
            ax.plot(x, [r[f"llm_reject_rate::{m}"] for r in t], "^:", lw=1.2, alpha=0.85,
                    label=f"LLM {m.split('/')[1]} (reject rate)")
        ax.set_title({"weird": "WEIRD arm", "smallscale": "Small-scale arm"}[arm])
        ax.set_xlabel("offer (units of 100)")
        ax.set_xticks(OFFERS)
        ax.set_ylim(-0.02, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("rejection")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Responder rejection: decision model vs attribute sigmoid vs LLM agents (same profiles)")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    res = [json.loads(l) for l in open(OUT)] if args.analyze_only else run()
    summary = analyze(res)
    ANALYSIS.write_text(json.dumps(summary, indent=2))
    plot(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cell_table"}, indent=2))
    for r in summary["cell_table"]:
        llm = "  ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in r.items() if k.startswith("llm_"))
        print(f"{r['arm']:>10} off{r['offer']:<3} jev={r['jev_mean_p_reject']:.3f} "
              f"sig={r['sigmoid_mean_p_reject']:.3f}  {llm}")
