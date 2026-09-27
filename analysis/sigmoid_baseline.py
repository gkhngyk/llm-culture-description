"""Complexity substrate 1.2 — sigmoid diagnostic baseline.

Classical-ABM ablation: predict rejection probability from
(fairness_salience, offer_fraction) WITHOUT consulting an LLM.
Compared against the LLM-driven decision for every pair — the
differential is the GABM-vs-ABM delta reported in §7.
"""

from __future__ import annotations

import math
from typing import Iterable


def sigmoid(x: float, *, threshold: float = 0.0, steepness: float = 8.0) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (x - threshold)))


def sigmoid_reject_probability(fairness_salience: float, offer_fraction: float) -> float:
    """Higher fairness_salience and lower offer_fraction -> higher reject prob.

    Parameter choice: threshold at zero so the rule is 'reject if fairness
    exceeds offer', scaled by steepness=8. Intentionally simple; this is a
    baseline to differ from, not a tuned model.
    """
    return sigmoid(fairness_salience - offer_fraction, threshold=0.0, steepness=8.0)


def compare_llm_vs_sigmoid(rows: Iterable[dict]) -> dict:
    """Return per-row (llm_reject, sigmoid_reject_prob) and aggregate agreement."""
    llm_rate = 0
    sig_sum = 0.0
    n = 0
    disagreements = 0
    for r in rows:
        resp_profile = r["responder_profile"]
        fs = float(resp_profile["fairness_salience"])
        offer_frac = r["offer"] / 100.0
        sp = sigmoid_reject_probability(fs, offer_frac)
        llm_reject = int(r["rejected"])
        sig_reject = int(sp >= 0.5)
        if llm_reject != sig_reject:
            disagreements += 1
        llm_rate += llm_reject
        sig_sum += sp
        n += 1
    return {
        "n": n,
        "llm_mean_reject": llm_rate / n if n else float("nan"),
        "sigmoid_mean_reject_prob": sig_sum / n if n else float("nan"),
        "point_agreement": 1.0 - (disagreements / n if n else 0.0),
    }
