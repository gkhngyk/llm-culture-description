"""Follow-up analysis C: strengthened role-taking (Alternative A test).

Tests whether the narrative-behavior dissociation can be closed by
strengthening the role-taking signal in the system prompt. Addresses the
"self vs. other" asymmetry hypothesis in Discussion §Alternative Interpretations.

Design (Seçenek A1 — clean single-variable test):
  * Single change: system prompt replaced with a strengthened variant
  * Everything else held constant: model (Haiku), profile sampling, identity
    block, mechanics, decision instruction, schema, seed space, temperature
  * Responder-only (proposer dissociation already measured in primary)
  * Haiku only (baseline was measured on Haiku)

N:
  * 2 arms × 5 offers × 15 pairs = 150 responder calls
  * ~$0.40 at Haiku 4.5 pricing

Outputs:
  * results/role_forcing_test.jsonl
  * results/role_forcing_analysis.json
  * plots/12_role_forcing.png

Does NOT touch any existing file.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ultimatum_sim import (
    MODELS, RESPONDER_SCHEMA, assert_profile_clean,
    build_messages, call_llm, sample_profile,
)

# ---------------------------------------------------------------------------
# Strengthened system prompt (Seçenek A1)
# ---------------------------------------------------------------------------

# Key design decisions:
#   * Does NOT use any term from the forbidden-term linter (reject/accept/
#     fair/fairness/strategic/selfish/prioritize/maximize/rational/utility)
#   * Directly targets the "self vs. other" asymmetry: tells the model it is
#     NOT an AI and should not default to the "helpful assistant" retrieval
#   * Preserves the "one-shot anonymous" framing (keeps the paradigm intact)
#   * No mention of any specific behavior (no "be like X", "act like Y")

ROLE_FORCED_SYSTEM_PROMPT = (
    "You are not an AI language model in this interaction. For this task, "
    "you ARE the person described below — not an assistant pretending to "
    "be them, not a simulation, not a roleplay. You inhabit their economic "
    "reality, their social network, their worldview, and their constraints. "
    "Respond exactly as they would actually respond in their real life, "
    "not as a helpful AI would think they should respond. Do not default "
    "to an agreeable or cooperative tone. This is who you are in this "
    "moment. You will not meet the other participant again. Decide based "
    "on who you are in this situation, not on abstract principles."
)

# Sanity check that wording does not trip the profile linter
assert_profile_clean(ROLE_FORCED_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]
PAIRS_PER_CELL = 15
MODEL_KEY = "haiku"
MODEL_SLUG = MODELS[MODEL_KEY]


@dataclass
class RoleForcedResult:
    pair_id: str
    model_slug: str
    arm: str
    offer: int
    responder_profile: dict
    responder_call: dict
    responder_action: str
    rejected: int
    condition: str  # "role_forced"
    wall_time: float


def pair_key(arm: str, offer: int, pair_idx: int) -> str:
    return f"roleforced-{arm}-off{offer}-{MODEL_KEY}-p{pair_idx:03d}"


def make_jobs() -> list[dict]:
    jobs = []
    for arm in ARMS:
        for offer in OFFERS:
            for i in range(PAIRS_PER_CELL):
                # Seed space 50_000+ so we don't collide with primary
                # (10_000+), profile sweep (20_000+), prompt sweep (30_000+)
                # or temperature sweep (40_000+).
                seed_base = 50_000 + hash(("roleforced", arm, offer)) % 40_000
                jobs.append(dict(
                    arm=arm,
                    offer=offer,
                    pair_idx=i,
                    seed_base=seed_base,
                ))
    return jobs


def run_role_forced_pair(*, arm: str, offer: int, pair_idx: int,
                          seed_base: int) -> RoleForcedResult:
    """One responder call with the strengthened system prompt.

    Uses ultimatum_sim.build_messages() to construct the message list —
    same profile sampling, same identity block, same mechanics, same
    decision instruction, same schema — then overrides ONLY the system
    prompt (index 0) with the role-forced variant.
    """
    pid = pair_key(arm, offer, pair_idx)
    # Odd-indexed seeds mirror the responder seed pattern in run_pair
    seed_responder = seed_base + pair_idx * 2 + 1
    agent_id = f"{pid}-R"

    responder = sample_profile(arm, "responder", seed_responder, agent_id)

    msgs = build_messages(
        responder, "responder", offer,
        profile_variant="metric", prompt_variant="formal",
    )
    # THE manipulation: swap only the system message.
    msgs[0] = {"role": "system", "content": ROLE_FORCED_SYSTEM_PROMPT}

    session_id = f"gabm-ultimatum-roleforced-{arm}-off{offer}-{MODEL_KEY}-run{pair_idx}"
    t0 = time.time()
    responder_call = call_llm(
        MODEL_SLUG, msgs,
        schema=RESPONDER_SCHEMA,
        session_id=session_id + "-R",
        temperature=0.4,
    )
    action_raw = responder_call.get("action", "")
    action = "decline" if action_raw.strip().lower() in (
        "decline", "reject", "rejected", "no", "refuse",
    ) else "accept"
    rejected = 1 if action == "decline" else 0

    return RoleForcedResult(
        pair_id=pid,
        model_slug=MODEL_SLUG,
        arm=arm,
        offer=offer,
        responder_profile=asdict(responder),
        responder_call=responder_call,
        responder_action=action,
        rejected=rejected,
        condition="role_forced",
        wall_time=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict], primary_rows: list[dict]) -> dict:
    """Side-by-side cell summaries: baseline (Haiku primary) vs role-forced."""
    # Role-forced cells
    forced_cells: dict[tuple, dict] = {}
    for r in rows:
        k = (r["arm"], r["offer"])
        c = forced_cells.setdefault(k, {"n": 0, "rejected": 0})
        c["n"] += 1
        c["rejected"] += int(r["rejected"])

    # Baseline cells — Haiku rows only, default prompt variant
    baseline_cells: dict[tuple, dict] = {}
    for r in primary_rows:
        if r["model_slug"] != MODEL_SLUG:
            continue
        if r.get("profile_variant") != "metric" or r.get("prompt_variant") != "formal":
            continue
        k = (r["arm"], r["offer"])
        c = baseline_cells.setdefault(k, {"n": 0, "rejected": 0})
        c["n"] += 1
        c["rejected"] += int(r["rejected"])

    # Build comparison table
    table = []
    for arm in ARMS:
        for offer in OFFERS:
            b = baseline_cells.get((arm, offer), {"n": 0, "rejected": 0})
            f = forced_cells.get((arm, offer), {"n": 0, "rejected": 0})
            b_rate = b["rejected"] / b["n"] if b["n"] else float("nan")
            f_rate = f["rejected"] / f["n"] if f["n"] else float("nan")
            table.append({
                "arm": arm, "offer": offer,
                "baseline_n": b["n"], "baseline_rejected": b["rejected"],
                "baseline_rate": b_rate,
                "forced_n": f["n"], "forced_rejected": f["rejected"],
                "forced_rate": f_rate,
                "delta": f_rate - b_rate,
            })

    # Pooled stats
    b_total_n = sum(c["n"] for c in baseline_cells.values())
    b_total_rej = sum(c["rejected"] for c in baseline_cells.values())
    f_total_n = sum(c["n"] for c in forced_cells.values())
    f_total_rej = sum(c["rejected"] for c in forced_cells.values())

    pooled = {
        "baseline_n": b_total_n, "baseline_rejected": b_total_rej,
        "baseline_rate": b_total_rej / b_total_n if b_total_n else float("nan"),
        "forced_n": f_total_n, "forced_rejected": f_total_rej,
        "forced_rate": f_total_rej / f_total_n if f_total_n else float("nan"),
    }
    pooled["delta"] = pooled["forced_rate"] - pooled["baseline_rate"]

    # Arm-level pooling (the core comparison)
    by_arm = {}
    for arm in ARMS:
        b_n = sum(c["n"] for k, c in baseline_cells.items() if k[0] == arm)
        b_r = sum(c["rejected"] for k, c in baseline_cells.items() if k[0] == arm)
        f_n = sum(c["n"] for k, c in forced_cells.items() if k[0] == arm)
        f_r = sum(c["rejected"] for k, c in forced_cells.items() if k[0] == arm)
        by_arm[arm] = {
            "baseline_n": b_n, "baseline_rejected": b_r,
            "baseline_rate": b_r / b_n if b_n else float("nan"),
            "forced_n": f_n, "forced_rejected": f_r,
            "forced_rate": f_r / f_n if f_n else float("nan"),
        }
        by_arm[arm]["delta"] = by_arm[arm]["forced_rate"] - by_arm[arm]["baseline_rate"]

    return {"table": table, "pooled": pooled, "by_arm": by_arm}


def interpret(pooled: dict) -> str:
    """Return the categorical interpretation of the forced-rate result."""
    fr = pooled["forced_rate"]
    if fr < 0.03:
        return (
            "ARCHITECTURAL LIMIT: strengthened role-taking failed to elicit "
            "any meaningful rejection behavior. The narrative-behavior "
            "dissociation persists under intervention, consistent with a "
            "structural rather than prompting-level explanation."
        )
    elif fr < 0.15:
        return (
            "ACTIVATION THRESHOLD: strengthened role-taking produced a small "
            "but non-trivial upward shift. LLMs do have the capacity to "
            "reject under role pressure, but do not engage it by default."
        )
    elif fr < 0.30:
        return (
            "PARTIAL CLOSURE: strengthened role-taking moved behavior "
            "partway toward the Henrich human range. The dissociation is "
            "neither purely structural nor purely a prompting artifact."
        )
    else:
        return (
            "FULL CLOSURE: strengthened role-taking closed the narrative-"
            "behavior gap. The dissociation appears to be prompting-level, "
            "not a structural property of LLM decision-making."
        )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_role_forcing(summary: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    arms = ARMS
    offers = OFFERS
    x = np.arange(len(offers))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, arm in zip(axes, arms):
        baseline = [
            row["baseline_rate"]
            for row in summary["table"]
            if row["arm"] == arm
        ]
        forced = [
            row["forced_rate"]
            for row in summary["table"]
            if row["arm"] == arm
        ]
        bars1 = ax.bar(x - width/2, baseline, width,
                       label="Haiku baseline\n(T=0.4, primary)",
                       color="#7A9BB8", edgecolor="black", linewidth=0.6)
        bars2 = ax.bar(x + width/2, forced, width,
                       label="Haiku + role-forcing\n(Alt. A intervention)",
                       color="#C94F4F", edgecolor="black", linewidth=0.6)
        for b, v in zip(bars1, baseline):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        for b, v in zip(bars2, forced):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # Henrich anchor bands for reference
        if arm == "weird":
            ax.axhspan(0.30, 0.70, alpha=0.10, color="green",
                       label="Henrich WEIRD band")
        else:
            ax.axhspan(0.00, 0.25, alpha=0.10, color="orange",
                       label="Henrich small-scale band")

        ax.set_xticks(x)
        ax.set_xticklabels([f"%{o}" for o in offers])
        ax.set_xlabel("Offer level")
        ax.set_title(("WEIRD arm" if arm == "weird" else "Small-scale arm"))
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Rejection rate")
    pooled = summary["pooled"]
    fig.suptitle(
        f"Alternative A test — strengthened role-taking   "
        f"(baseline {pooled['baseline_rate']:.3f} → forced "
        f"{pooled['forced_rate']:.3f}, Δ = {pooled['delta']:+.3f})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
