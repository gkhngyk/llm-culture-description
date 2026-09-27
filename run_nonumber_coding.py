"""
Theme coding of Study 2 (no-number) responder reasonings with the same
7-theme codebook and the same two decision-model coders as the Study 1
coding (run_decision_model_coding.py). Analysis only; no new experiment.

  python run_nonumber_coding.py run
Outputs: results/nonumber_coding.jsonl, results/nonumber_coding_analysis.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq
from scipy import stats

import run_decision_model_coding as dm

RESULTS = Path(__file__).parent / "results"
OUT = RESULTS / "nonumber_coding.jsonl"
ANALYSIS = RESULTS / "nonumber_coding_analysis.json"


def load_texts() -> list[dict]:
    texts = []
    for l in open(RESULTS / "nonumber.jsonl"):
        r = json.loads(l)
        texts.append({"corpus": "study2", "tid": f"study2-{r['model_key']}-{r['pair_id']}",
                      "arm": r["arm"], "offer": r["offer"], "action": r["responder_action"],
                      "source_model": r["model_slug"], "text": r["responder_call"]["reasoning"]})
    return texts


def run() -> None:
    texts = load_texts()
    done = set()
    if OUT.exists():
        done = {(d["tid"], d["coder"], d["repeat"]) for d in map(json.loads, open(OUT))}
    todo = [j for j in dm._jobs(texts) if (j[0]["tid"], j[1], j[2]) not in done]
    print(f"{len(done)} cached, {len(todo)} calls", flush=True)
    with ThreadPoolExecutor(8) as ex, open(OUT, "a") as f:
        for fut in as_completed([ex.submit(dm.code_text, *j) for j in todo]):
            f.write(json.dumps(fut.result()) + "\n"); f.flush()


def analyze() -> dict:
    rows = [json.loads(l) for l in open(OUT)]
    agg = defaultdict(lambda: defaultdict(list)); meta = {}
    for r in rows:
        meta[r["tid"]] = r
        for k, v in r["p"].items():
            agg[(r["tid"], r["coder"])][k].append(v)
    P = {key: {k: float(np.mean(v)) for k, v in d.items()} for key, d in agg.items()}
    tids = sorted(meta); themes = list(dm.THEMES)
    vec = lambda c, th, sel: np.array([P[(t, c)][th] for t in sel])
    out = {"n_texts": len(tids), "n_calls": len(rows),
           "cost_usd": round(sum((r["usage"] or {}).get("cost", 0) for r in rows), 5)}
    sep = {}
    for coder in (dm.SPAN, dm.JEV):
        for th in themes:
            w = vec(coder, th, [t for t in tids if meta[t]["arm"] == "weird"])
            s = vec(coder, th, [t for t in tids if meta[t]["arm"] == "smallscale"])
            sep[f"{coder}|{th}"] = {"weird_prev": round(float(np.mean(w >= .5)), 3),
                                    "smallscale_prev": round(float(np.mean(s >= .5)), 3),
                                    "auc": round(float(stats.mannwhitneyu(w, s).statistic / (len(w) * len(s))), 3)}
    out["arm_separation"] = sep
    y = np.array([meta[t]["arm"] == "weird" for t in tids], float)
    acc = {}
    for coder in (dm.SPAN, dm.JEV):
        X = np.column_stack([vec(coder, th, tids) for th in themes] + [np.ones(len(tids))])
        hits = 0
        for i in range(len(tids)):
            m = np.ones(len(tids), bool); m[i] = False
            hits += int((X[i] @ lstsq(X[m], y[m], rcond=None)[0] >= .5) == bool(y[i]))
        acc[coder] = round(hits / len(tids), 3)
    out["loo_arm_accuracy"] = acc
    # Which themes go with refusal, within arm (does reasoning match the decision?)
    rej = {}
    for coder in (dm.SPAN, dm.JEV):
        for th in themes:
            a = vec(coder, th, [t for t in tids if meta[t]["action"] == "decline"])
            b = vec(coder, th, [t for t in tids if meta[t]["action"] == "accept"])
            rej[f"{coder}|{th}"] = {"prev_decline": round(float(np.mean(a >= .5)), 3),
                                    "prev_accept": round(float(np.mean(b >= .5)), 3)}
    out["theme_by_decision"] = rej
    return out


if __name__ == "__main__":
    run()
    s = analyze()
    ANALYSIS.write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=1))
