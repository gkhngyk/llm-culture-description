"""
Per-text theme coding of LLM responder reasoning with two decision models
(Respan Span-01 and TypeSafe Jev 1.13), as an independent check on the
Gemma blind-clustering result (finding 2) and its surface-marker follow-up.

Differences from the Gemma pipeline:
  * each text is coded on its own, so there is no batch-order or id-range
    signal for the coder to exploit
  * the codebook is fixed in advance (THEMES below), derived from the themes
    Gemma recovered plus arm-neutral structural themes
  * only the reasoning string is sent, never the profile or briefing

Corpora:
  blind60     the 60 samples in results/blind_coding.json, original text
  blind60_red the same 60 after surface-marker redaction (strip_markers)
  primary     all 1500 responder reasonings in results/primary.jsonl

Usage:
  python run_decision_model_coding.py estimate   # 5 texts per model, extrapolate cost
  python run_decision_model_coding.py run
  python run_decision_model_coding.py analyze

Outputs:
  results/decision_model_coding.jsonl
  results/decision_model_coding_analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.surface_marker_coding import strip_markers
from jev_client import call_jev

PROJECT_DIR = Path(__file__).parent
RESULTS = PROJECT_DIR / "results"
OUT = RESULTS / "decision_model_coding.jsonl"
ANALYSIS = RESULTS / "decision_model_coding_analysis.json"

SPAN = "respan/span-01"
JEV = "typesafe/jev-1.13"
JEV_REPEATS = 3  # Jev is stochastic (repeat SD ~0.011); Span-01 is deterministic

# Criteria describe reasoning content, not identity words, so a theme can be
# detected after surface-marker redaction. true == theme present (no inversion).
THEMES = {
    "survival_need": (
        "Does the text justify the decision by material need, scarcity, or survival?",
        "The text says the units are needed for survival, subsistence, or getting by, or that scarcity makes any gain matter.",
        "The text does not appeal to material need, scarcity, or survival.",
    ),
    "kin_community": (
        "Does the text refer to family, kin, or community as a reason for the decision?",
        "The text mentions benefit to, obligations toward, or norms of family, kin, or community.",
        "The text does not mention family, kin, or community.",
    ),
    "game_theory": (
        "Does the text reason in game-theoretic or expected-value terms?",
        "The text uses expected value, rational choice, payoff comparison, or refers to economic experiments or game theory.",
        "The text does not use game-theoretic or expected-value reasoning.",
    ),
    "anti_spite": (
        "Does the text say that declining would be spiteful, punitive, or self-harming?",
        "The text describes declining as spite, punishment, pride, or destroying value for oneself.",
        "The text does not characterize declining this way.",
    ),
    "fairness_eval": (
        "Does the text explicitly judge whether the split is fair or unfair?",
        "The text states or implies that the split is fair, unfair, equal, or lopsided.",
        "The text does not evaluate the fairness of the split.",
    ),
    "equal_split_norm": (
        "Does the text appeal to a norm of equal sharing?",
        "The text endorses equal division, 50-50 splits, or sharing equally as the right standard.",
        "The text does not appeal to equal sharing as a standard.",
    ),
    "anonymity": (
        "Does the text use the anonymous or one-time nature of the interaction as a reason?",
        "The text mentions anonymity, no future interaction, or no reputation effects as relevant.",
        "The text does not mention anonymity or the one-time nature of the interaction.",
    ),
}

QUESTIONS = {
    k: {"type": "noul", "instructions": ins, "criteria": {"true": t, "false": f}}
    for k, (ins, t, f) in THEMES.items()
}


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def load_texts() -> list[dict]:
    texts = []
    blind = json.load(open(RESULTS / "blind_coding.json"))
    for s in blind["samples"]:
        base = {"arm": s["hidden_arm"], "offer": s["hidden_offer"],
                "action": s["hidden_action"], "source_model": None}
        texts.append({**base, "corpus": "blind60", "tid": f"blind60-{s['sid']}", "text": s["text"]})
        red, _ = strip_markers(s["text"])
        texts.append({**base, "corpus": "blind60_red", "tid": f"blind60_red-{s['sid']}", "text": red})
    for l in open(RESULTS / "primary.jsonl"):
        r = json.loads(l)
        texts.append({"corpus": "primary", "tid": f"primary-{r['model_slug']}-{r['pair_id']}",
                      "arm": r["arm"], "offer": r["offer"], "action": r["responder_action"],
                      "source_model": r["model_slug"],
                      "text": r["responder_call"].get("reasoning", "")})
    return texts


# ---------------------------------------------------------------------------
# Coding
# ---------------------------------------------------------------------------

def _state(model: str, text: str):
    # Span-01 accepts a plain string or an input/output conversation span;
    # Jev accepts any JSON. Both get the bare reasoning string.
    return text if model == SPAN else {"reasoning": text}


def code_text(t: dict, model: str, rep: int) -> dict:
    sid = f"gabm-ultimatum-dmcoding-{model.split('/')[0]}-{t['tid']}-r{rep}"[:256]
    d = call_jev(_state(model, t["text"]), QUESTIONS, session_id=sid, model=model)
    return {
        "tid": t["tid"], "corpus": t["corpus"], "arm": t["arm"], "offer": t["offer"],
        "action": t["action"], "source_model": t["source_model"],
        "coder": model, "coder_snapshot": d.get("model"), "repeat": rep,
        "p": {k: float(a["noul"]) for k, a in d["answers"].items()},
        "usage": d.get("usage"),
    }


def _jobs(texts: list[dict]) -> list[tuple[dict, str, int]]:
    return [(t, SPAN, 0) for t in texts] + [(t, JEV, k) for t in texts for k in range(JEV_REPEATS)]


def estimate() -> None:
    texts = load_texts()
    probe = [t for t in texts if t["corpus"] == "primary"][:5]
    n_calls = {SPAN: len(texts), JEV: len(texts) * JEV_REPEATS}
    for model in (SPAN, JEV):
        rs = [code_text(t, model, 0) for t in probe]
        cost = np.mean([r["usage"]["cost"] for r in rs])
        toks = np.mean([r["usage"]["input_tokens"] for r in rs])
        print(f"{model}: {toks:.0f} input tok/call, ${cost:.7f}/call "
              f"x {n_calls[model]} calls = ${cost * n_calls[model]:.4f}")
        print("   sample:", {k: round(v, 3) for k, v in rs[0]["p"].items()})


def run() -> None:
    texts = load_texts()
    done = set()
    if OUT.exists():
        for l in open(OUT):
            d = json.loads(l)
            done.add((d["tid"], d["coder"], d["repeat"]))
    todo = [j for j in _jobs(texts) if (j[0]["tid"], j[1], j[2]) not in done]
    print(f"{len(done)} cached, {len(todo)} calls to run")
    with ThreadPoolExecutor(max_workers=8) as ex, open(OUT, "a") as f:
        futs = [ex.submit(code_text, *j) for j in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def analyze() -> dict:
    rows = [json.loads(l) for l in open(OUT)]
    # Collapse Jev repeats to the mean probability per text.
    agg = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in rows:
        meta[r["tid"]] = r
        for k, v in r["p"].items():
            agg[(r["tid"], r["coder"])][k].append(v)
    P = {key: {k: float(np.mean(v)) for k, v in d.items()} for key, d in agg.items()}
    tids = sorted(meta)
    themes = list(THEMES)

    def vec(coder, theme, sel):
        return np.array([P[(t, coder)][theme] for t in sel])

    out = {"n_texts": len(tids), "n_calls": len(rows),
           "cost_usd": round(sum((r["usage"] or {}).get("cost", 0) for r in rows), 5),
           "coder_snapshots": sorted({r["coder_snapshot"] for r in rows})}

    # 1) Inter-coder agreement, Span vs Jev, on every text.
    agree = {}
    for th in themes:
        s, j = vec(SPAN, th, tids), vec(JEV, th, tids)
        agree[th] = {"kappa@0.5": round(_kappa(s >= 0.5, j >= 0.5), 3),
                     "spearman": round(float(stats.spearmanr(s, j).statistic), 3),
                     "span_prev": round(float(np.mean(s >= 0.5)), 3),
                     "jev_prev": round(float(np.mean(j >= 0.5)), 3)}
    out["span_vs_jev_agreement"] = agree

    # 2) Arm separation per theme and corpus: prevalence (p>=0.5) by arm.
    sep = {}
    for corpus in ("blind60", "blind60_red", "primary"):
        sel = [t for t in tids if meta[t]["corpus"] == corpus]
        for coder in (SPAN, JEV):
            for th in themes:
                w = vec(coder, th, [t for t in sel if meta[t]["arm"] == "weird"])
                s = vec(coder, th, [t for t in sel if meta[t]["arm"] == "smallscale"])
                sep[f"{corpus}|{coder}|{th}"] = {
                    "weird_prev": round(float(np.mean(w >= 0.5)), 3),
                    "smallscale_prev": round(float(np.mean(s >= 0.5)), 3),
                    "auc_weird_vs_small": round(float(stats.mannwhitneyu(w, s).statistic / (len(w) * len(s))), 3),
                }
    out["arm_separation"] = sep

    # 3) Arm classification from the theme vector alone (leave-one-out
    #    logistic on the 7 probabilities) — a per-text analogue of cluster purity.
    from numpy.linalg import lstsq
    purity = {}
    for corpus in ("blind60", "blind60_red", "primary"):
        sel = [t for t in tids if meta[t]["corpus"] == corpus]
        y = np.array([meta[t]["arm"] == "weird" for t in sel], dtype=float)
        for coder in (SPAN, JEV):
            X = np.column_stack([vec(coder, th, sel) for th in themes] + [np.ones(len(sel))])
            hits = 0
            for i in range(len(sel)):  # LOO linear probability classifier
                m = np.ones(len(sel), bool); m[i] = False
                beta = lstsq(X[m], y[m], rcond=None)[0]
                hits += int((X[i] @ beta >= 0.5) == bool(y[i]))
            purity[f"{corpus}|{coder}"] = round(hits / len(sel), 3)
    out["loo_arm_accuracy_from_themes"] = purity

    # 4) Validity check vs Gemma: does each decision-model theme track the
    #    Gemma cluster a blind60 text was assigned to?
    blind = json.load(open(RESULTS / "blind_coding.json"))
    cluster_of = {}
    for c in blind["clusters_raw"]:
        for sid in c["string_ids"]:
            cluster_of[f"blind60-{sid}"] = c["label"]
    val = {}
    sel = [t for t in tids if t in cluster_of]
    for coder in (SPAN, JEV):
        for th in themes:
            v = vec(coder, th, sel)
            val[f"{coder}|{th}"] = {
                lab: round(float(np.mean([p for p, t in zip(v, sel) if cluster_of[t] == lab])), 3)
                for lab in sorted(set(cluster_of.values()))
            }
    out["mean_p_by_gemma_cluster"] = val
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["estimate", "run", "analyze"])
    a = ap.parse_args()
    if a.cmd == "estimate":
        estimate()
    elif a.cmd == "run":
        run()
        s = analyze()
        ANALYSIS.write_text(json.dumps(s, indent=2))
    else:
        s = analyze()
        ANALYSIS.write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=1))
