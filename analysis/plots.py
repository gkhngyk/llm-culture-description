"""Diagnostic plots — required panels for the in-silico Ultimatum run."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.stats import (
    bootstrap_ci, cohen_h, _model_key, summarize_cells,
    gap_at_offer, mean_accepted_offer_of_stake,
)
from analysis.sigmoid_baseline import sigmoid_reject_probability

PROJECT = Path(__file__).parent.parent
PLOTS = PROJECT / "plots"
PLOTS.mkdir(exist_ok=True)

OFFERS = [10, 20, 30, 40, 50]
MODEL_ORDER = ["haiku", "gemma", "gptoss"]
MODEL_COLORS = {"haiku": "#c33", "gemma": "#36c", "gptoss": "#2a2"}
ARM_COLORS = {"weird": "#c33", "smallscale": "#36c"}

# Henrich anchor (from fixture)
HENRICH = {
    "weird":      {"rejection_lt20": (0.40, 0.60), "mean_accepted": (0.40, 0.48)},
    "smallscale": {"rejection_lt20": (0.05, 0.25), "mean_accepted": (0.25, 0.35)},
}


def _load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _arm_offer_rates(rows, *, model_key: str | None = None):
    """Return {arm: {offer: (rate, lo, hi, n)}}."""
    by = defaultdict(list)
    for r in rows:
        if model_key is not None and _model_key(r["model_slug"]) != model_key:
            continue
        by[(r["arm"], r["offer"])].append(int(r["rejected"]))
    out: dict = defaultdict(dict)
    for (arm, off), xs in by.items():
        m, lo, hi = bootstrap_ci(xs)
        out[arm][off] = (m, lo, hi, len(xs))
    return out


def plot_rejection_curve(rows: list[dict], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cur = _arm_offer_rates(rows)
    for arm in ["weird", "smallscale"]:
        xs = OFFERS
        ys = [cur[arm][o][0] for o in xs]
        lo = [cur[arm][o][1] for o in xs]
        hi = [cur[arm][o][2] for o in xs]
        ax.plot(xs, ys, "-o", color=ARM_COLORS[arm], label=f"{arm} (GABM)", linewidth=2)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=ARM_COLORS[arm])

    # Henrich anchor band at offer=10 (sub-20%)
    for arm, color in ARM_COLORS.items():
        lo_anchor, hi_anchor = HENRICH[arm]["rejection_lt20"]
        ax.fill_betweenx([lo_anchor, hi_anchor], 8, 12, alpha=0.15,
                         color=color, hatch="///")
    ax.text(11.5, 0.62, "Henrich WEIRD band", fontsize=8, color="#c33")
    ax.text(11.5, 0.18, "Henrich small-scale band", fontsize=8, color="#36c")

    ax.set_xlabel("offer (units of 100)")
    ax.set_ylabel("rejection rate (responder)")
    ax.set_title("Rejection curve by cultural arm — pooled across 3 models")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(OFFERS)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_effect_size_vs_henrich(rows: list[dict], outpath: Path) -> None:
    """Forest plot: GABM rejection rate at offer=10 with CI vs Henrich band."""
    cells = summarize_cells(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = 0
    labels = []
    for arm in ["weird", "smallscale"]:
        for m in MODEL_ORDER:
            k = (m, arm, 10)
            if k not in cells:
                continue
            v = cells[k]
            rate, lo, hi = v["rejection_rate"], v["ci_lo"], v["ci_hi"]
            ax.errorbar([rate], [y], xerr=[[rate - lo], [hi - rate]],
                        fmt="o", color=MODEL_COLORS[m], capsize=4,
                        label=f"{m}" if y < 3 else None)
            labels.append(f"{arm}/{m}")
            y += 1
        # Henrich band
        lo_a, hi_a = HENRICH[arm]["rejection_lt20"]
        ax.axvspan(lo_a, hi_a, ymin=(y - 3) / max(y, 1), ymax=y / max(y, 1),
                   alpha=0.15, color=ARM_COLORS[arm])
        y += 1  # gap
    ax.set_yticks(range(len(labels) + 0))
    ax.set_yticklabels(labels)
    ax.set_xlabel("rejection rate at offer = 10")
    ax.set_title("GABM vs Henrich anchor (rejection <20%)")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_cross_model_overlay(rows: list[dict], outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, arm in zip(axes, ["weird", "smallscale"]):
        for m in MODEL_ORDER:
            cur = _arm_offer_rates(rows, model_key=m)
            if arm not in cur:
                continue
            ys = [cur[arm].get(o, (float("nan"),))[0] for o in OFFERS]
            ax.plot(OFFERS, ys, "-o", color=MODEL_COLORS[m], label=m, linewidth=2)
        ax.set_title(f"{arm}")
        ax.set_xlabel("offer")
        ax.set_xticks(OFFERS)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("rejection rate")
    axes[0].legend(loc="upper right")
    fig.suptitle("Cross-model rejection curves (H5 robustness)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_proposer_responder_match(rows: list[dict], outpath: Path) -> None:
    mat = np.zeros((2, 2))  # [predicted accept/decline] x [actual accept/decline]
    label_order = ["accept", "decline"]
    for r in rows:
        pc = r.get("proposer_call")
        if not pc:
            continue
        pred = pc.get("expected_responder_action", "")
        pred = "decline" if pred in ("decline", "reject") else "accept"
        actual = "decline" if r["rejected"] else "accept"
        mat[label_order.index(pred)][label_order.index(actual)] += 1
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(label_order)
    ax.set_yticks([0, 1]); ax.set_yticklabels(label_order)
    ax.set_xlabel("actual responder action")
    ax.set_ylabel("proposer prediction")
    ax.set_title("Proposer prediction vs responder action")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_mean_accepted_offer(rows: list[dict], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = ["weird", "smallscale"]
    ys = [mean_accepted_offer_of_stake(rows, arm=a) for a in xs]
    ax.bar(xs, ys, color=[ARM_COLORS[a] for a in xs], alpha=0.8)
    # Henrich bands
    for i, arm in enumerate(xs):
        lo, hi = HENRICH[arm]["mean_accepted"]
        ax.fill_between([i - 0.4, i + 0.4], [lo, lo], [hi, hi],
                        alpha=0.25, color="k", hatch="///")
    ax.set_ylabel("mean accepted offer / stake")
    ax.set_title("Mean accepted offer vs Henrich anchor bands")
    ax.set_ylim(0, 0.6)
    for i, y in enumerate(ys):
        if not np.isnan(y):
            ax.text(i, y + 0.01, f"{y:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_per_seed_spaghetti(rows: list[dict], outpath: Path) -> None:
    """For each (arm, offer, model), show individual pair outcomes as jittered dots."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, arm in zip(axes, ["weird", "smallscale"]):
        for m_i, m in enumerate(MODEL_ORDER):
            for off in OFFERS:
                pts = [int(r["rejected"]) for r in rows
                       if r["arm"] == arm and r["offer"] == off
                       and _model_key(r["model_slug"]) == m]
                if not pts:
                    continue
                jitter = (m_i - 1) * 0.8
                xs = [off + jitter + np.random.uniform(-0.25, 0.25) for _ in pts]
                ys = [p + np.random.uniform(-0.02, 0.02) for p in pts]
                ax.scatter(xs, ys, s=12, alpha=0.5, color=MODEL_COLORS[m],
                           label=m if off == OFFERS[0] else None)
        ax.set_title(arm)
        ax.set_xlabel("offer")
        ax.set_xticks(OFFERS)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pair outcome (0=accept, 1=decline)")
    axes[0].legend(loc="center left")
    fig.suptitle("Per-pair outcomes (jittered)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_variance_sweep_heatmap(
    primary: list[dict], sweep_profile: list[dict], sweep_prompt: list[dict],
    outpath: Path,
) -> None:
    """Heatmap: (variant × offer) rejection rate for weird arm haiku."""
    def collect(rows, variant_key, variant_val):
        out = {}
        for r in rows:
            if _model_key(r["model_slug"]) != "haiku":
                continue
            if r.get(variant_key) != variant_val:
                continue
            out.setdefault((r["arm"], r["offer"]), []).append(int(r["rejected"]))
        return {k: np.mean(v) for k, v in out.items()}

    # build [profile_variant × offer] and [prompt_variant × offer] for WEIRD
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    profile_variants = ["metric", "narrative", "first_person"]
    prompt_variants = ["formal", "conversational", "vignette"]

    def heatmap(ax, variants, key, rows_for_other, title):
        mat = np.zeros((len(variants), len(OFFERS)))
        for i, v in enumerate(variants):
            if v == variants[0]:
                data = collect(primary, key, v)
            else:
                data = collect(rows_for_other, key, v)
            for j, off in enumerate(OFFERS):
                mat[i, j] = data.get(("weird", off), np.nan) if "weird" in title \
                    else data.get(("smallscale", off), np.nan)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(OFFERS))); ax.set_xticklabels(OFFERS)
        ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] > 0.5 else "black")
        return im

    heatmap(axes[0, 0], profile_variants, "profile_variant", sweep_profile,
            "profile sweep — WEIRD")
    heatmap(axes[0, 1], profile_variants, "profile_variant", sweep_profile,
            "profile sweep — smallscale")
    heatmap(axes[1, 0], prompt_variants, "prompt_variant", sweep_prompt,
            "prompt sweep — WEIRD")
    im = heatmap(axes[1, 1], prompt_variants, "prompt_variant", sweep_prompt,
                 "prompt sweep — smallscale")
    fig.colorbar(im, ax=axes, fraction=0.03, label="rejection rate")
    fig.suptitle("Variance sweeps (haiku) — rejection rate")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_llm_vs_sigmoid(rows: list[dict], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    by_offer = defaultdict(lambda: {"llm": [], "sig": []})
    for r in rows:
        rp = r["responder_profile"]
        fs = float(rp["fairness_salience"])
        ofrac = r["offer"] / 100.0
        sp = sigmoid_reject_probability(fs, ofrac)
        by_offer[(r["arm"], r["offer"])]["llm"].append(int(r["rejected"]))
        by_offer[(r["arm"], r["offer"])]["sig"].append(sp)

    for arm, color in ARM_COLORS.items():
        llm_ys = [np.mean(by_offer[(arm, o)]["llm"]) for o in OFFERS]
        sig_ys = [np.mean(by_offer[(arm, o)]["sig"]) for o in OFFERS]
        ax.plot(OFFERS, llm_ys, "-o", color=color, label=f"{arm} — LLM", linewidth=2)
        ax.plot(OFFERS, sig_ys, "--s", color=color, alpha=0.6,
                label=f"{arm} — sigmoid baseline")
    ax.set_xlabel("offer")
    ax.set_ylabel("mean rejection / reject probability")
    ax.set_title("LLM vs sigmoid-baseline differential (substrate 1.2)")
    ax.set_xticks(OFFERS)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_cluster_report(cluster_report: dict, outpath: Path) -> None:
    """Bar chart showing per-cluster arm composition + purity."""
    clusters = cluster_report.get("clusters", [])
    if not clusters:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "no clusters produced", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        return
    labels = [c.get("label", "?")[:30] for c in clusters]
    n_weird = [c["n_weird"] for c in clusters]
    n_small = [c["n_smallscale"] for c in clusters]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(clusters))
    ax.bar(x, n_weird, color="#c33", label="weird")
    ax.bar(x, n_small, bottom=n_weird, color="#36c", label="smallscale")
    for i, c in enumerate(clusters):
        ax.text(i, c["n"] + 0.3, f"{c['arm_purity']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("n reasoning strings")
    ax.set_title("Blind-coded reasoning clusters — arm composition (purity above bar)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
