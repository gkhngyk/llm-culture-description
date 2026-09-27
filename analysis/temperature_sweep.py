"""Follow-up analysis: temperature robustness sweep.

Tests whether the near-zero rejection rate observed at T=0.4 in the primary
run is robust across sampling temperatures. Addresses limitation #1 in the
paper.

Design (wide):
  * 3 models × 2 arms × 5 offers × 2 new temperatures × 15 pairs
  * Temperatures tested: {0.0, 1.0}  (T=0.4 already in primary)
  * Total: 900 pairs = 1800 role calls

Outputs:
  * results/temperature_sweep.jsonl
  * results/temperature_sweep_analysis.json
  * plots/10_temperature_sweep.png

Does NOT touch any existing files or rerun any primary calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ultimatum_sim import MODELS

OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]
NEW_TEMPERATURES = [0.0, 1.0]  # primary was 0.4
PAIRS_PER_CELL = 15

MODEL_KEYS = ["haiku", "gemma", "gptoss"]


def make_temperature_jobs() -> list[dict]:
    """Produce the full job list for the temperature sweep."""
    jobs: list[dict] = []
    for model_key in MODEL_KEYS:
        slug = MODELS[model_key]
        for temp in NEW_TEMPERATURES:
            for arm in ARMS:
                for offer in OFFERS:
                    for i in range(PAIRS_PER_CELL):
                        # Distinct seed space (40_000+) so we never collide
                        # with primary (10_000+), profile (20_000+), prompt
                        # (30_000+) seeds.
                        seed_base = 40_000 + hash(
                            (model_key, f"t{temp}", arm, offer)
                        ) % 50_000
                        jobs.append(dict(
                            model_slug=slug,
                            model_key=model_key,
                            arm=arm,
                            offer=offer,
                            pair_idx=i,
                            profile_variant="metric",
                            prompt_variant="formal",
                            temperature=temp,
                            run_proposer=False,  # responder-only, cuts cost 2x
                            seed_base=seed_base,
                        ))
    return jobs


def pair_key(j: dict) -> str:
    """Unique key for a temperature-sweep pair; stable across reruns."""
    t_tag = f"t{j['temperature']:.1f}".replace(".", "p")
    return (f"temp-{t_tag}-{j['arm']}-off{j['offer']}"
            f"-{j['model_key']}-p{j['pair_idx']:03d}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_sweep(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    """Compute rejection rate by (model, temperature, arm, offer) cell."""
    cells: dict[tuple, dict] = {}
    for r in rows:
        # Recover temperature from the stored temperature field; fall back
        # to parsing from the pair_id if missing.
        temp = _row_temperature(r)
        model_key = _row_model_key(r["model_slug"])
        key = (model_key, round(temp, 2), r["arm"], r["offer"])
        c = cells.setdefault(key, {"n": 0, "rejected": 0})
        c["n"] += 1
        c["rejected"] += int(r["rejected"])
    out = {}
    for (m, t, a, o), c in cells.items():
        rate = c["rejected"] / c["n"] if c["n"] else float("nan")
        out[f"{m}|t{t:.1f}|{a}|{o}"] = {
            "model": m, "temperature": t, "arm": a, "offer": o,
            "n": c["n"], "rejected": c["rejected"], "rate": rate,
        }
    return out


def pooled_by_temperature(cells: dict) -> dict:
    """Pool across models and offers to get temperature-level rejection."""
    by_t_arm: dict[tuple, dict] = {}
    for c in cells.values():
        k = (round(c["temperature"], 2), c["arm"])
        b = by_t_arm.setdefault(k, {"n": 0, "rejected": 0})
        b["n"] += c["n"]
        b["rejected"] += c["rejected"]
    return {
        f"t{t:.1f}|{a}": {
            "temperature": t, "arm": a, "n": v["n"],
            "rejected": v["rejected"],
            "rate": v["rejected"] / v["n"] if v["n"] else float("nan"),
        }
        for (t, a), v in by_t_arm.items()
    }


def _row_temperature(r: dict) -> float:
    # pair_id encodes temperature; parse "temp-t0p0-..." style
    pid = r.get("pair_id", "")
    if pid.startswith("temp-t"):
        tok = pid.split("-")[1]  # e.g. "t0p0"
        try:
            return float(tok[1:].replace("p", "."))
        except Exception:
            pass
    # fallback to stored field if present (newer runs)
    return float(r.get("temperature", r.get("_temperature", 0.4)))


def _row_model_key(slug: str) -> str:
    for k, s in MODELS.items():
        if s == slug:
            return k
    return "unknown"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_temperature_sweep(rows: list[dict], primary_rows: list[dict],
                            out_path: Path) -> None:
    """Side-by-side bar chart: rejection rate by (temperature, arm)."""
    cells = summarize(rows)
    pooled = pooled_by_temperature(cells)

    # Add primary T=0.4 point from existing primary data.
    primary_cells: dict[tuple, dict] = {}
    for r in primary_rows:
        key = (r["arm"],)
        c = primary_cells.setdefault(key, {"n": 0, "rejected": 0})
        c["n"] += 1
        c["rejected"] += int(r["rejected"])

    t_values = sorted({v["temperature"] for v in pooled.values()} | {0.4})
    arms = ["weird", "smallscale"]
    colors = {"weird": "#C06C84", "smallscale": "#6C5B7B"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(t_values))
    width = 0.38

    for i, arm in enumerate(arms):
        rates = []
        for t in t_values:
            if abs(t - 0.4) < 1e-6:
                c = primary_cells.get((arm,), {"n": 0, "rejected": 0})
                rates.append(c["rejected"] / c["n"] if c["n"] else 0.0)
            else:
                k = f"t{t:.1f}|{arm}"
                rates.append(pooled.get(k, {"rate": 0.0})["rate"])
        offset = (-1 if i == 0 else 1) * width / 2
        bars = ax.bar(x + offset, rates, width,
                      label=("WEIRD" if arm == "weird" else "Small-scale"),
                      color=colors[arm], edgecolor="black", linewidth=0.6)
        for b, rate in zip(bars, rates):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                    f"{rate:.3f}", ha="center", va="bottom", fontsize=8)

    # Henrich anchor band
    ax.axhspan(0.30, 0.70, alpha=0.12, color="green",
               label="Henrich WEIRD anchor")
    ax.axhspan(0.00, 0.25, alpha=0.12, color="orange",
               label="Henrich small-scale anchor")

    ax.set_xticks(x)
    ax.set_xticklabels([f"T={t:.1f}" for t in t_values])
    ax.set_ylabel("Rejection rate (pooled across models & offers)")
    ax.set_xlabel("Sampling temperature")
    ax.set_title("Temperature robustness sweep — pooled rejection rates")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    primary_marker = "(primary)"
    ax.text(list(t_values).index(0.4), -0.07, primary_marker,
            ha="center", va="top", fontsize=8, style="italic",
            transform=ax.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
