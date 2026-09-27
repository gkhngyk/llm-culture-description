"""Complexity substrate 1.6 — emergence detection on the rejection curve.

Phase-transition test: look for sharp jumps in rejection rate across offer
levels (within each arm × model) that exceed a z-threshold over the
rolling baseline. Zero findings is a valid result.
"""

from __future__ import annotations

import numpy as np


def detect_rejection_phase_transitions(
    curve_by_offer: dict[int, float],
    *,
    phase_transition_z: float = 2.5,
) -> list[dict]:
    offers = sorted(curve_by_offer.keys())
    series = [curve_by_offer[o] for o in offers]
    findings = []
    for i in range(1, len(series)):
        delta = abs(series[i] - series[i - 1])
        baseline = float(np.std(series[:i])) if i > 2 else delta
        if baseline > 0 and delta > baseline * phase_transition_z:
            findings.append({
                "type": "PHASE_TRANSITION",
                "between_offers": (offers[i - 1], offers[i]),
                "delta": delta,
                "baseline_std": baseline,
            })
    return findings


def detect_curve_monotonicity_break(curve_by_offer: dict[int, float]) -> list[dict]:
    """Flag any non-monotone step (rejection rate going UP with higher offer)."""
    offers = sorted(curve_by_offer.keys())
    findings = []
    for i in range(1, len(offers)):
        delta = curve_by_offer[offers[i]] - curve_by_offer[offers[i - 1]]
        if delta > 1e-9:
            findings.append({
                "type": "MONOTONICITY_BREAK",
                "between_offers": (offers[i - 1], offers[i]),
                "delta": delta,
            })
    return findings
