"""Complexity substrate 1.6-b — blind mechanism coding.

Sample reasoning strings from the primary run (arm + offer labels redacted),
ship them to a second LLM (different provider from the generator), and ask
it to cluster them into qualitative themes. Reveal labels only after
clustering is complete.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ultimatum_sim import call_llm, MODELS

CODER_MODEL = MODELS["gemma"]  # different provider from haiku primary


CLUSTER_SCHEMA = {
    "type": "object",
    "required": ["clusters"],
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "theme", "string_ids"],
                "properties": {
                    "label": {"type": "string"},
                    "theme": {"type": "string"},
                    "string_ids": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    },
}


def sample_reasoning(primary_rows: list[dict], *, per_cell: int = 3, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_cell: dict[tuple, list[dict]] = {}
    for r in primary_rows:
        if r["model_slug"] != MODELS["haiku"]:
            continue
        k = (r["arm"], r["offer"])
        by_cell.setdefault(k, []).append(r)
    samples = []
    idx = 0
    for k, rows in sorted(by_cell.items()):
        rng.shuffle(rows)
        for r in rows[:per_cell]:
            reasoning = r["responder_call"].get("reasoning", "")
            samples.append({
                "sid": idx,
                "text": reasoning,
                "hidden_arm": k[0],
                "hidden_offer": k[1],
                "hidden_action": r["responder_action"],
            })
            idx += 1
    return samples


def run_blind_clustering(samples: list[dict]) -> dict:
    """Ask the coder LLM to cluster samples with arm/offer redacted."""
    redacted = [{"id": s["sid"], "text": s["text"]} for s in samples]
    system = (
        "You are a qualitative researcher coding free-text reasoning from "
        "an anonymous behavioral experiment. Do not infer which condition "
        "any respondent was in — cluster strings purely by the themes you "
        "observe in the text."
    )
    user = (
        "Below are decision-reasoning strings from participants. "
        "Cluster them into 3 to 6 qualitative themes. For each cluster, "
        "give a short label, a one-sentence theme description, and the "
        "list of string ids that belong to it. Every id must appear in "
        "exactly one cluster.\n\n"
        f"STRINGS (JSON):\n{json.dumps(redacted, ensure_ascii=False)}"
    )
    result = call_llm(
        CODER_MODEL,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema=CLUSTER_SCHEMA,
        session_id="gabm-ultimatum-blindcoding",
        temperature=0.3,
        max_tokens=2000,
    )
    return result


def reveal_and_score(samples: list[dict], clusters: list[dict]) -> dict:
    """Post-clustering reveal: for each cluster, measure arm purity."""
    by_sid = {s["sid"]: s for s in samples}
    out = []
    for c in clusters:
        ids = c.get("string_ids", [])
        arms = [by_sid[i]["hidden_arm"] for i in ids if i in by_sid]
        offers = [by_sid[i]["hidden_offer"] for i in ids if i in by_sid]
        n = len(arms)
        n_weird = sum(1 for a in arms if a == "weird")
        n_small = sum(1 for a in arms if a == "smallscale")
        purity = max(n_weird, n_small) / n if n else 0.0
        out.append({
            "label": c.get("label"),
            "theme": c.get("theme"),
            "n": n,
            "n_weird": n_weird,
            "n_smallscale": n_small,
            "arm_purity": purity,
            "mean_offer": sum(offers) / n if n else 0.0,
        })
    return {"clusters": out}
