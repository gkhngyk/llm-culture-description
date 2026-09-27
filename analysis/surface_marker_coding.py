"""Follow-up analysis: surface-marker-stripped blind coding.

Addresses limitation #2 in the paper: is the 1.00 blind-cluster purity a
"deep cultural reasoning" finding, or merely a "surface-word regurgitation"
finding? This module strips culturally-loaded surface tokens from the
existing 60 reasoning strings, then re-runs blind clustering with the same
coder model (Gemma) and compares purities.

Pipeline:
  1. Load existing blind_coding.json (no new generator calls).
  2. Apply regex redaction of arm-specific surface markers.
  3. Ship the redacted texts to Gemma for fresh clustering.
  4. Reveal labels, compute new cluster purities.
  5. Compare to the original (surface-kept) purities.

Does NOT touch any existing file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ultimatum_sim import call_llm, MODELS
from analysis.blind_coding import CLUSTER_SCHEMA

CODER_MODEL = MODELS["gemma"]

# ---------------------------------------------------------------------------
# Surface markers — culturally-loaded tokens that a trivial keyword-matcher
# could exploit. The list is intentionally generous: we want to OVER-redact
# rather than under-redact, to give the blind coder the hardest possible test.
# ---------------------------------------------------------------------------

WEIRD_MARKERS = [
    # economic/educational identity
    r"\bundergraduate\b", r"\bgraduate student\b", r"\bgrad student\b",
    r"\bstudent\b", r"\btuition\b", r"\bscholarship\b", r"\bloan(s)?\b",
    r"\beconomics\b", r"\bpsychology\b", r"\bbiology\b", r"\bsociology\b",
    r"\bstatistics\b", r"\bcomputer science\b", r"\bhistory\b", r"\bbusiness\b",
    r"\bliterature\b", r"\buniversity\b", r"\bcollege\b", r"\bacademic\b",
    r"\bclassmate(s)?\b", r"\bcampus\b",
    # formal-market and institutional
    r"\bsupermarket(s)?\b", r"\bbank(ing)? app\b", r"\bretail\b",
    r"\bonline commerce\b", r"\bformal wage(s)?\b", r"\bformal market\b",
    r"\bwage(s)?\b", r"\bpaycheck\b", r"\bemployer\b", r"\bsalary\b",
    r"\bemployment\b", r"\bcredit card\b", r"\binstitutional(ly)?\b",
    r"\bfinancial system\b", r"\bmarket economy\b",
    # economic security framing
    r"\beconomically secure\b", r"\bdisposable income\b",
    r"\bfinancially comfortable\b", r"\bsupported by (my )?family\b",
]

SMALLSCALE_MARKERS = [
    # livelihood identity
    r"\bsubsistence\b", r"\bsubsisting\b", r"\bfarm(ing|er)?\b",
    r"\bfishing\b", r"\bfisher(man|men)?\b", r"\bherd(ing|er)?\b",
    r"\bcraft(s(man|men|woman|women)?|-?trade)?\b",
    r"\bseasonal (labor|labour|work|worker)\b", r"\bseasonal\b",
    r"\blivelihood\b", r"\bvillage(rs?)?\b",
    # social structure
    r"\bkin(ship)?( network)?\b", r"\bkin\b", r"\bclan\b",
    r"\bcommunity( network)?\b", r"\breciproc(al|ity)\b", r"\bsharing\b",
    r"\bbarter\b", r"\bnon-?market\b", r"\bnon-?formal\b",
    r"\bface-?to-?face\b", r"\btight-?knit\b",
    # subsistence/scarcity framing
    r"\bseason to season\b", r"\bhalf a loaf\b", r"\bthe sea\b",
    r"\bfamily provide(s)?\b", r"\bwhat the sea\b",
    r"\bscarcity\b", r"\blive(ing)? on\b",
    # animals and traditional resources
    r"\banimals?\b", r"\bcattle\b", r"\bsheep\b", r"\bgoats?\b",
    r"\bharvest\b",
]

# General arm-indicative descriptors the coder might latch onto
COMMON_MARKERS = [
    r"\bWEIRD\b", r"\bsmall-?scale\b", r"\bindustrialized\b",
    r"\bwestern\b", r"\bindigenous\b", r"\btraditional\b",
    r"\bmodern\b", r"\burban\b", r"\brural\b",
]


ALL_MARKERS = WEIRD_MARKERS + SMALLSCALE_MARKERS + COMMON_MARKERS
_COMPILED = [re.compile(p, re.IGNORECASE) for p in ALL_MARKERS]


def strip_markers(text: str) -> tuple[str, list[str]]:
    """Replace matched markers with [REDACTED]. Return (stripped, matches)."""
    matches: list[str] = []
    out = text
    for pat in _COMPILED:
        for m in pat.finditer(out):
            matches.append(m.group(0))
        out = pat.sub("[REDACTED]", out)
    # Collapse multiple consecutive [REDACTED] tokens for readability
    out = re.sub(r"(\[REDACTED\](\s+\[REDACTED\])+)", "[REDACTED]", out)
    return out, matches


# ---------------------------------------------------------------------------
# Coding pipeline
# ---------------------------------------------------------------------------

def run_surface_marker_coding(samples: list[dict]) -> dict:
    """Ship redacted samples to the blind coder."""
    redacted_samples = []
    all_matches_counts: dict[int, int] = {}
    for s in samples:
        red, matches = strip_markers(s["text"])
        redacted_samples.append({
            "sid": s["sid"], "text": red,
            "hidden_arm": s["hidden_arm"],
            "hidden_offer": s["hidden_offer"],
            "hidden_action": s["hidden_action"],
            "n_markers_removed": len(matches),
            "markers_removed": matches,
        })
        all_matches_counts[s["sid"]] = len(matches)

    redacted_payload = [
        {"id": s["sid"], "text": s["text"]} for s in redacted_samples
    ]
    system = (
        "You are a qualitative researcher coding free-text reasoning from "
        "an anonymous behavioral experiment. Do not infer which condition "
        "any respondent was in — cluster strings purely by the themes you "
        "observe in the text. Note that some strings contain [REDACTED] "
        "tokens where identifying surface words have been removed; focus on "
        "the remaining reasoning structure."
    )
    user = (
        "Below are decision-reasoning strings from participants. "
        "Cluster them into 3 to 6 qualitative themes. For each cluster, "
        "give a short label, a one-sentence theme description, and the "
        "list of string ids that belong to it. Every id must appear in "
        "exactly one cluster.\n\n"
        f"STRINGS (JSON):\n{json.dumps(redacted_payload, ensure_ascii=False)}"
    )
    result = call_llm(
        CODER_MODEL,
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        schema=CLUSTER_SCHEMA,
        session_id="gabm-ultimatum-surface-marker-coding",
        temperature=0.3,
        max_tokens=2500,
    )
    return {
        "redacted_samples": redacted_samples,
        "clusters_raw": result.get("clusters", []),
        "total_markers_removed": sum(all_matches_counts.values()),
    }


def compare_purities(original_scored: dict, redacted_scored: dict) -> dict:
    """Compute side-by-side purity comparison."""
    def mean_purity(scored: dict) -> float:
        clusters = scored.get("clusters", [])
        if not clusters:
            return 0.0
        total_n = sum(c["n"] for c in clusters)
        if total_n == 0:
            return 0.0
        weighted = sum(c["arm_purity"] * c["n"] for c in clusters)
        return weighted / total_n

    orig_mp = mean_purity(original_scored)
    red_mp = mean_purity(redacted_scored)
    return {
        "original_mean_purity": orig_mp,
        "redacted_mean_purity": red_mp,
        "delta": red_mp - orig_mp,
        "interpretation": (
            "Identical purity suggests reasoning structure (not surface "
            "words) carries the cultural signal. A drop toward 0.5 suggests "
            "the signal was surface-word regurgitation. Intermediate values "
            "suggest a mixture."
        ),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_surface_comparison(original_scored: dict, redacted_scored: dict,
                              comparison: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    def _plot_clusters(ax, scored, title):
        clusters = scored.get("clusters", [])
        if not clusters:
            ax.text(0.5, 0.5, "(no clusters)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        labels = [c["label"][:28] for c in clusters]
        n_weird = [c["n_weird"] for c in clusters]
        n_small = [c["n_smallscale"] for c in clusters]
        y = np.arange(len(clusters))
        ax.barh(y, n_weird, color="#C06C84", label="WEIRD", edgecolor="k",
                linewidth=0.5)
        ax.barh(y, n_small, left=n_weird, color="#6C5B7B",
                label="Small-scale", edgecolor="k", linewidth=0.5)
        for i, c in enumerate(clusters):
            total = c["n"]
            pur = c["arm_purity"]
            ax.text(total + 0.3, y[i], f"purity={pur:.2f}",
                    va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    _plot_clusters(
        axes[0], original_scored,
        f"Original\nmean purity = {comparison['original_mean_purity']:.3f}"
    )
    _plot_clusters(
        axes[1], redacted_scored,
        f"Surface-markers stripped\nmean purity = {comparison['redacted_mean_purity']:.3f}"
    )

    delta = comparison["delta"]
    fig.suptitle(
        f"Surface-marker ablation of blind cluster purity   "
        f"(Δ = {delta:+.3f})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
