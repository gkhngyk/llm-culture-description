"""
Runner + analysis for the card wording test (prereg_card_wording.md).

  python run_card_wording.py preview   # print the three cards for one identity per arm, no API
  python run_card_wording.py smoke     # 1 identity per arm x offer, all wordings, all models (36 calls)
  python run_card_wording.py run       # N=50 (1 800 calls), resumable
  python run_card_wording.py analyze

Outputs: results/card_wording.jsonl, results/card_wording_smoke.jsonl,
         results/card_wording_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.stats import chi2_two_sample, cohen_h
from card_wording_test import WORDINGS, profile_seed, render_card, run_one
from ultimatum_sim import MODELS, sample_profile

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
OUT = RESULTS / "card_wording.jsonl"
SMOKE = RESULTS / "card_wording_smoke.jsonl"
ANALYSIS = RESULTS / "card_wording_analysis.json"

OFFERS = [10, 20]
ARMS = ["weird", "smallscale"]
N_PAIRS = 50
CONCURRENCY = 8


def _jobs(n: int) -> list[dict]:
    return [dict(model_key=mk, model_slug=slug, arm=a, offer=o, pair_idx=i, wording=w)
            for mk, slug in MODELS.items() for a in ARMS for o in OFFERS
            for i in range(n) for w in WORDINGS]


def _key(j):
    return f"{j['wording']}|{j['model_key']}|cw-{j['arm']}-off{j['offer']}-p{j['pair_idx']:03d}"


def _run(jobs, path: Path) -> None:
    done = {json.loads(l)["key"] for l in open(path) if l.strip()} if path.exists() else set()
    todo = [j for j in jobs if _key(j) not in done]
    print(f"[cardw] total={len(jobs)} done={len(done)} remaining={len(todo)}", flush=True)
    t0, ok, fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex, open(path, "a") as f:
        futs = {ex.submit(run_one, **j): j for j in todo}
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n"); f.flush(); ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {futs[fut]}: {e}", file=sys.stderr, flush=True)
    print(f"[cardw] done ok={ok} fail={fail} {time.time() - t0:.0f}s", flush=True)


def preview() -> None:
    for arm in ARMS:
        seed = profile_seed(arm, 10, 0)
        p = sample_profile(arm, "responder", seed, "preview")
        for w in WORDINGS:
            print(f"--- {arm} / {w} ---\n{render_card(p, w, seed)}")


def _classify(rs: list[dict], k: int) -> dict:
    w = [r["rejected"] for r in rs if r["arm"] == "weird"]
    s = [r["rejected"] for r in rs if r["arm"] == "smallscale"]
    pw, ps = float(np.mean(w)), float(np.mean(s))
    _, p = chi2_two_sample(sum(s), len(s), sum(w), len(w))
    p_bonf = min(1.0, p * k)
    h = cohen_h(ps, pw)
    gap = ps - pw
    if gap >= 0.15 and p_bonf < 0.01 and abs(h) >= 0.3:
        label = "reversed (stereotype direction)"
    elif gap <= -0.15 and p_bonf < 0.01 and abs(h) >= 0.3:
        label = "Henrich direction"
    elif abs(gap) < 0.05:
        label = "no gap"
    else:
        label = "inconclusive"
    return {"weird": round(pw, 4), "smallscale": round(ps, 4), "gap_S_minus_W": round(gap, 4),
            "n_w": len(w), "n_s": len(s), "p": p, "p_bonf": p_bonf, "cohen_h": round(h, 3),
            "label": label}


def analyze(path: Path = OUT) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out: dict = {"n_rows": len(rows), "invalid_payloads": sum(r["n_invalid_payloads"] for r in rows)}
    prim = {w: _classify([r for r in rows if r["wording"] == w and r["offer"] == 10], k=2)
            for w in ("minimal", "ethnographic")}
    out["primary_offer10_pooled"] = prim
    labels = {w: prim[w]["label"] for w in prim}
    rev = {w: labels[w].startswith("reversed") for w in labels}
    if labels["ethnographic"] == "Henrich direction":
        reading = "Henrich direction under ethnographic cards (central finding)"
    elif all(rev.values()):
        reading = "reversal robust to wording: stereotype interpretation supported"
    elif any(rev.values()):
        reading = "wording-dependent: claim narrowed to " + ", ".join(w for w in rev if rev[w])
    else:
        reading = "reversal produced by original wording: title/headline must be revised"
    out["overall_reading"] = reading
    out["original_offer10_pooled"] = _classify([r for r in rows if r["wording"] == "original" and r["offer"] == 10], k=1)
    out["per_model_offer10"] = {f"{mk}|{w}": _classify([r for r in rows if r["model_key"] == mk and r["wording"] == w and r["offer"] == 10], k=1)
                                for mk in MODELS for w in WORDINGS}
    out["offer20_pooled"] = {w: _classify([r for r in rows if r["wording"] == w and r["offer"] == 20], k=1) for w in WORDINGS}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preview", "smoke", "run", "analyze"])
    a = ap.parse_args()
    if a.cmd == "preview":
        preview()
    elif a.cmd == "smoke":
        _run(_jobs(1), SMOKE)
    else:
        if a.cmd == "run":
            _run(_jobs(N_PAIRS), OUT)
        s = analyze(OUT)
        ANALYSIS.write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=1))
