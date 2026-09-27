"""Statistical tests for the Ultimatum in-silico replication.

All tests are pre-registered. Nothing here picks the sign of the effect.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

BOOT = 10_000
RNG = np.random.default_rng(7)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group(
    rows: list[dict],
    *,
    key_fn,
) -> dict[tuple, list[int]]:
    """Return {key -> list of rejected 0/1}."""
    out: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        out[key_fn(r)].append(int(r["rejected"]))
    return out


def bootstrap_ci(xs: list[int], n_boot: int = BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(xs.mean())
    idx = RNG.integers(0, len(xs), size=(n_boot, len(xs)))
    resamples = xs[idx].mean(axis=1)
    lo = float(np.quantile(resamples, alpha / 2))
    hi = float(np.quantile(resamples, 1 - alpha / 2))
    return mean, lo, hi


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    def phi(p):
        p = min(max(p, 1e-9), 1 - 1e-9)
        return 2 * math.asin(math.sqrt(p))
    return phi(p1) - phi(p2)


def chi2_two_sample(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-sample chi-square on 2x2 contingency. Returns (chi2, p_two_sided)."""
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    if table.sum() == 0:
        return (0.0, 1.0)
    try:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        return float(chi2), float(p)
    except ValueError:
        return (0.0, 1.0)


def spearman_within(arm_rows: list[tuple[int, float]]) -> tuple[float, float]:
    """arm_rows = [(offer, rejection_rate), ...]  — returns (rho, p)."""
    xs = [r[0] for r in arm_rows]
    ys = [r[1] for r in arm_rows]
    res = stats.spearmanr(xs, ys)
    return float(res.statistic), float(res.pvalue)


def summarize_cells(rows: list[dict]) -> dict:
    """Return per (model, arm, offer) cell: n, rejection_count, rate, CI."""
    by = defaultdict(list)
    for r in rows:
        model_slug = r["model_slug"]
        model_key = _model_key(model_slug)
        by[(model_key, r["arm"], r["offer"])].append(int(r["rejected"]))
    out = {}
    for k, xs in by.items():
        mean, lo, hi = bootstrap_ci(xs)
        out[k] = {
            "n": len(xs),
            "rejection_count": int(sum(xs)),
            "acceptance_count": int(len(xs) - sum(xs)),
            "rejection_rate": mean,
            "ci_lo": lo,
            "ci_hi": hi,
        }
    return out


def _model_key(slug: str) -> str:
    if "haiku" in slug:
        return "haiku"
    if "gemma" in slug:
        return "gemma"
    if "gpt-oss" in slug or "gpt_oss" in slug:
        return "gptoss"
    return slug


def gap_at_offer(
    cells: dict, offer: int, model_key: str | None = None
) -> dict:
    """Compute WEIRD − small-scale rejection gap, with chi2 p-value and Cohen h.
    If model_key is None, pools across models.
    """
    def pool(arm: str):
        xs = 0
        ns = 0
        for (m, a, o), v in cells.items():
            if o != offer or a != arm:
                continue
            if model_key is not None and m != model_key:
                continue
            xs += v["rejection_count"]
            ns += v["n"]
        return xs, ns
    wx, wn = pool("weird")
    sx, sn = pool("smallscale")
    wr = wx / wn if wn else float("nan")
    sr = sx / sn if sn else float("nan")
    chi2, p = chi2_two_sample(wx, wn, sx, sn)
    h = cohen_h(wr, sr) if (wn and sn) else float("nan")
    return {
        "offer": offer, "model": model_key,
        "weird_rate": wr, "weird_n": wn,
        "small_rate": sr, "small_n": sn,
        "gap": wr - sr, "chi2": chi2, "p": p,
        "cohen_h": h,
    }


def bonferroni(p_values: list[float], k: int | None = None) -> list[float]:
    k = k or len(p_values)
    return [min(1.0, p * k) for p in p_values]


def mean_accepted_offer_of_stake(rows: list[dict], *, arm: str, model_key: str | None = None) -> float:
    accepted_offers = []
    for r in rows:
        if r["arm"] != arm:
            continue
        if model_key is not None and _model_key(r["model_slug"]) != model_key:
            continue
        if r["rejected"] == 0:
            accepted_offers.append(r["offer"])
    if not accepted_offers:
        return float("nan")
    return float(np.mean(accepted_offers)) / 100.0
