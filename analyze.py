"""End-to-end analysis + plotting + baseline gate + report generation.

Run AFTER primary + sweeps + blind coding have finished.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analysis import plots
from analysis.stats import (
    _model_key, bonferroni, cohen_h, gap_at_offer,
    load_jsonl, mean_accepted_offer_of_stake, spearman_within, summarize_cells,
)
from analysis.sigmoid_baseline import compare_llm_vs_sigmoid
from analysis.emergence import (
    detect_rejection_phase_transitions, detect_curve_monotonicity_break,
)

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS = PROJECT / "plots"

OFFERS = [10, 20, 30, 40, 50]
MODEL_KEYS = ["haiku", "gemma", "gptoss"]

# Baseline-gate thresholds (from fixture)
GATE = {
    "weird":      {"rejection_lt20_range": (0.30, 0.70),
                    "rejection_4050_max": 0.15,
                    "mean_accepted_range": (0.30, 0.55)},
    "smallscale": {"rejection_lt20_range": (0.00, 0.40)},
    "gap_min": 0.15,
    "gap_p_max": 0.01,
    "effect_size_min": 0.3,
}


def _rate(rows, *, arm, offer, model_key=None):
    xs = []
    for r in rows:
        if r["arm"] != arm or r["offer"] != offer:
            continue
        if model_key is not None and _model_key(r["model_slug"]) != model_key:
            continue
        xs.append(int(r["rejected"]))
    return (sum(xs) / len(xs)) if xs else float("nan"), len(xs)


def main() -> None:
    primary = load_jsonl(RESULTS / "primary.jsonl")
    sweep_profile = load_jsonl(RESULTS / "sweep_profile.jsonl") if (RESULTS / "sweep_profile.jsonl").exists() else []
    sweep_prompt = load_jsonl(RESULTS / "sweep_prompt.jsonl") if (RESULTS / "sweep_prompt.jsonl").exists() else []

    print(f"primary rows: {len(primary)}  sweep_profile: {len(sweep_profile)}  sweep_prompt: {len(sweep_prompt)}")

    # ------------- 1. cell summaries -------------
    cells = summarize_cells(primary)
    cells_ser = {f"{m}|{a}|{o}": v for (m, a, o), v in cells.items()}

    # ------------- 2. cross-model pooled rejection curve -------------
    pooled_curve = {}
    for arm in ["weird", "smallscale"]:
        for off in OFFERS:
            rate, n = _rate(primary, arm=arm, offer=off)
            pooled_curve[(arm, off)] = {"rate": rate, "n": n}

    # ------------- 3. between-arm gap at each offer level -------------
    gaps = []
    pvals = []
    for off in OFFERS:
        g = gap_at_offer(cells, off)
        gaps.append(g)
        pvals.append(g["p"])
    p_bonf = bonferroni(pvals, k=5)
    for g, pb in zip(gaps, p_bonf):
        g["p_bonferroni"] = pb

    # ------------- 4. Spearman monotonicity (H4) -------------
    spearman = {}
    for arm in ["weird", "smallscale"]:
        arm_rows = [(o, pooled_curve[(arm, o)]["rate"]) for o in OFFERS]
        rho, p = spearman_within(arm_rows)
        spearman[arm] = {"rho": rho, "p": p}

    # ------------- 5. cross-model divergence (H5) -------------
    h5 = []
    for arm in ["weird", "smallscale"]:
        for off in OFFERS:
            rates = {}
            for mk in MODEL_KEYS:
                rate, _ = _rate(primary, arm=arm, offer=off, model_key=mk)
                rates[mk] = rate
            mean = float(np.mean([v for v in rates.values() if not np.isnan(v)]))
            max_dev = max(abs(v - mean) for v in rates.values() if not np.isnan(v))
            h5.append({
                "arm": arm, "offer": off, "rates": rates,
                "mean": mean, "max_abs_dev": max_dev,
            })

    # ------------- 6. mean accepted offer per arm per model -------------
    mean_accepted = {}
    for arm in ["weird", "smallscale"]:
        mean_accepted[arm] = {
            "pooled": mean_accepted_offer_of_stake(primary, arm=arm),
        }
        for mk in MODEL_KEYS:
            mean_accepted[arm][mk] = mean_accepted_offer_of_stake(primary, arm=arm, model_key=mk)

    # ------------- 7. sigmoid baseline (substrate 1.2) -------------
    sig_compare = compare_llm_vs_sigmoid(primary)

    # ------------- 8. emergence detection (substrate 1.6) -------------
    emergence_findings = {}
    for arm in ["weird", "smallscale"]:
        curve = {o: pooled_curve[(arm, o)]["rate"] for o in OFFERS}
        pt = detect_rejection_phase_transitions(curve)
        mb = detect_curve_monotonicity_break(curve)
        emergence_findings[arm] = {"phase_transitions": pt, "monotonicity_breaks": mb}

    # ------------- 9. BASELINE GATE -------------
    weird10 = pooled_curve[("weird", 10)]["rate"]
    small10 = pooled_curve[("smallscale", 10)]["rate"]
    gap10 = gaps[0]
    gate_verdict = {
        "H1_weird_lt20_in_range": GATE["weird"]["rejection_lt20_range"][0] <= weird10 <= GATE["weird"]["rejection_lt20_range"][1],
        "H2_small_lt20_in_range": GATE["smallscale"]["rejection_lt20_range"][0] <= small10 <= GATE["smallscale"]["rejection_lt20_range"][1],
        "H3_gap_ge_0.15":         (weird10 - small10) >= GATE["gap_min"],
        "H3_p_lt_0.01_bonf":      gap10["p_bonferroni"] < GATE["gap_p_max"],
        "H3_cohen_h_ge_0.3":      abs(gap10["cohen_h"]) >= GATE["effect_size_min"],
        "H4_spearman_weird":      spearman["weird"]["rho"] > 0.9,
        "H4_spearman_small":      spearman["smallscale"]["rho"] > 0.9,
        "H5_all_cells_within_0.15": all(h["max_abs_dev"] <= 0.15 for h in h5),
    }
    finding_primary = all([
        gate_verdict["H1_weird_lt20_in_range"],
        gate_verdict["H2_small_lt20_in_range"],
        gate_verdict["H3_gap_ge_0.15"],
        gate_verdict["H3_p_lt_0.01_bonf"],
    ])

    out = {
        "n_primary": len(primary),
        "cells": cells_ser,
        "pooled_curve": {f"{a}|{o}": v for (a, o), v in pooled_curve.items()},
        "gaps_per_offer": gaps,
        "spearman": spearman,
        "h5_cross_model": h5,
        "mean_accepted_of_stake": mean_accepted,
        "sigmoid_comparison": sig_compare,
        "emergence": emergence_findings,
        "gate_verdict": gate_verdict,
        "finding_primary": finding_primary,
    }
    with open(RESULTS / "analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[gate] finding={finding_primary}  verdict={gate_verdict}")

    # ------------- plots -------------
    plots.plot_rejection_curve(primary, PLOTS / "1_rejection_curve.png")
    plots.plot_effect_size_vs_henrich(primary, PLOTS / "2_effect_size_vs_henrich.png")
    plots.plot_cross_model_overlay(primary, PLOTS / "3_cross_model_overlay.png")
    plots.plot_proposer_responder_match(primary, PLOTS / "4_proposer_responder_match.png")
    plots.plot_mean_accepted_offer(primary, PLOTS / "5_mean_accepted_offer.png")
    plots.plot_per_seed_spaghetti(primary, PLOTS / "6_per_seed_spaghetti.png")
    if sweep_profile or sweep_prompt:
        plots.plot_variance_sweep_heatmap(
            primary, sweep_profile, sweep_prompt,
            PLOTS / "7_variance_sweep_heatmap.png",
        )
    plots.plot_llm_vs_sigmoid(primary, PLOTS / "8_llm_vs_sigmoid_differential.png")
    # cluster dendrogram is generated by run_blind_coding.py post-hoc
    print("[plots] written to", PLOTS)


if __name__ == "__main__":
    main()
