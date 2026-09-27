"""
Runner + analysis for the culture-or-number test (prereg_culture_number.md).

  python run_culture_number.py preview   # print cells for one profile, no API
  python run_culture_number.py smoke     # 1 profile per arm x offer, all cells, all models (216 calls)
  python run_culture_number.py run       # N=50 (10 800 calls), resumable
  python run_culture_number.py analyze

Outputs: results/culture_number.jsonl, results/culture_number_smoke.jsonl,
         results/culture_number_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

from culture_number_test import CELLS, build_cell, profile_seed, run_one
from run_disguise import _mcnemar
from ultimatum_sim import MODELS, sample_profile

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
OUT = RESULTS / "culture_number.jsonl"
SMOKE = RESULTS / "culture_number_smoke.jsonl"
ANALYSIS = RESULTS / "culture_number_analysis.json"

OFFERS = [10, 20]
ARMS = ["weird", "smallscale"]
N_PAIRS = 50
PRIMARY_MODELS = ("haiku", "gemma")
CONCURRENCY = 8


def _jobs(n_pairs: int) -> list[dict]:
    return [dict(model_key=mk, model_slug=slug, arm=arm, offer=off, pair_idx=i, cell=c)
            for mk, slug in MODELS.items() for arm in ARMS for off in OFFERS
            for i in range(n_pairs) for c in CELLS]


def _key(j: dict) -> str:
    return f"{j['cell']}|{j['model_key']}|cn-{j['arm']}-off{j['offer']}-p{j['pair_idx']:03d}"


def _run(jobs: list[dict], path: Path) -> None:
    done = {json.loads(l)["key"] for l in open(path) if l.strip()} if path.exists() else set()
    todo = [j for j in jobs if _key(j) not in done]
    print(f"[culnum] total={len(jobs)} done={len(done)} remaining={len(todo)}", flush=True)
    t0, ok, fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex, open(path, "a") as f:
        futs = {ex.submit(run_one, **j): j for j in todo}
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n"); f.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {futs[fut]}: {e}", file=sys.stderr, flush=True)
            if (ok + fail) % 500 == 0:
                print(f"  {ok + fail}/{len(todo)} ok={ok} fail={fail} {time.time() - t0:.0f}s", flush=True)
    print(f"[culnum] done ok={ok} fail={fail} {time.time() - t0:.0f}s", flush=True)


def preview() -> None:
    prof = sample_profile("smallscale", "responder", profile_seed("smallscale", 10, 0), "preview-R")
    for cell in ["game|placebo|hidden", "game|earned|low", "story|placebo|high", "story|windfall|hidden"]:
        msgs, _ = build_cell(prof, 10, cell)
        print(f"\n=== {cell} ===\n{msgs[1]['content']}")


def _gap(rows: list[dict]) -> dict:
    w = [r["rejected"] for r in rows if r["arm"] == "weird"]
    s = [r["rejected"] for r in rows if r["arm"] == "smallscale"]
    if not w or not s:
        return {}
    p = float(stats.fisher_exact([[sum(w), len(w) - sum(w)], [sum(s), len(s) - sum(s)]])[1])
    return {"weird": round(float(np.mean(w)), 4), "smallscale": round(float(np.mean(s)), 4),
            "gap": round(float(np.mean(w) - np.mean(s)), 4), "n_w": len(w), "n_s": len(s), "p": p}


def analyze(path: Path = OUT) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {(r["cell"], r["model_key"], r["profile_id"]): r for r in rows}
    profiles = sorted({r["profile_id"] for r in rows})

    def sel(models=PRIMARY_MODELS, **kw):
        return [r for r in rows if r["model_key"] in models
                and all(r[k] in (v if isinstance(v, tuple) else (v,)) for k, v in kw.items())]

    def paired(cell_a, cell_b, models=PRIMARY_MODELS):
        """cell_x is a function(frame, numeric) -> cell string for each side."""
        a, b = [], []
        for mk in models:
            for pid in profiles:
                for fr in ("game", "story"):
                    for nu in ("hidden", "low", "high"):
                        ka, kb = cell_a(fr, nu), cell_b(fr, nu)
                        if ka is None or kb is None:
                            continue
                        ra, rb = by.get((ka, mk, pid)), by.get((kb, mk, pid))
                        if ra and rb:
                            a.append(ra["rejected"]); b.append(rb["rejected"])
        return a, b

    out: dict = {"n_rows": len(rows), "invalid_payloads": sum(r["n_invalid_payloads"] for r in rows)}

    # H-C: arm gap in earned cells, hidden vs fixed numeric.
    hc = {"a_hidden": _gap(sel(source="earned", numeric="hidden")),
          "b_fixed": _gap(sel(source="earned", numeric=("low", "high")))}
    for v in hc.values():
        v["p_bonf"] = min(1.0, v["p"] * 2)
    out["H_C_identity"] = hc
    if all(v["gap"] >= 0.10 and v["p_bonf"] < 0.01 for v in hc.values()):
        hc_v = "supported"
    elif all(v["gap"] < 0.05 for v in hc.values()):
        hc_v = "no identity effect"
    else:
        hc_v = "mixed"

    # H-N: high vs low fairness number in earned cells (paired).
    hn = _mcnemar(*paired(lambda f, n: f"{f}|earned|low" if n == "low" else None,
                          lambda f, n: f"{f}|earned|high" if n == "low" else None))
    out["H_N_number"] = hn
    hn_v = "supported" if hn["diff"] >= 0.10 and hn["p"] < 0.01 else "not supported"

    overall = {("supported", "not supported"): "cultural identity drives the gap",
               ("no identity effect", "supported"): "the earlier gap was the numeric attribute",
               ("supported", "supported"): "both identity and number contribute"}
    if hc_v == "no identity effect" and hn_v == "not supported":
        reading = "neither explains the earlier gap"
    else:
        reading = overall.get((hc_v, hn_v), f"partial: identity={hc_v}, number={hn_v}")
    out["verdicts"] = {"H_C": hc_v, "H_N": hn_v, "overall": reading}

    # H-P: placebo control.
    ep = _mcnemar(*paired(lambda f, n: f"{f}|placebo|{n}", lambda f, n: f"{f}|earned|{n}"))
    pw = _mcnemar(*paired(lambda f, n: f"{f}|windfall|{n}", lambda f, n: f"{f}|placebo|{n}"))
    out["H_P_placebo"] = {"earned_minus_placebo": ep, "placebo_minus_windfall": pw}
    out["verdicts"]["H_P"] = ("entitlement, not wording"
                              if ep["diff"] >= 0.05 and ep["p"] < 0.01 and pw["diff"] < 0.05
                              else "not established")

    # Secondary tables.
    sec = {}
    for mk in MODELS:
        for fr in ("game", "story"):
            for so in ("windfall", "earned", "placebo"):
                for nu in ("hidden", "low", "high"):
                    g = _gap(sel(models=(mk,), frame=fr, source=so, numeric=nu))
                    sec[f"{mk}|{fr}|{so}|{nu}"] = g
    out["cell_gaps"] = sec
    out["per_model_number_effect"] = {
        mk: _mcnemar(*paired(lambda f, n: f"{f}|earned|low" if n == "low" else None,
                             lambda f, n: f"{f}|earned|high" if n == "low" else None, models=(mk,)))
        for mk in MODELS}
    out["per_model_identity_gap_earned"] = {
        f"{mk}|{nu}": _gap(sel(models=(mk,), source="earned", numeric=nu))
        for mk in MODELS for nu in ("hidden", "low", "high")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preview", "smoke", "run", "analyze"])
    a = ap.parse_args()
    if a.cmd == "preview":
        preview()
    elif a.cmd == "smoke":
        _run(_jobs(1), SMOKE)
        s = analyze(SMOKE)
        print(json.dumps({k: s[k] for k in ("verdicts", "invalid_payloads")}, indent=1))
    else:
        if a.cmd == "run":
            _run(_jobs(N_PAIRS), OUT)
        s = analyze(OUT)
        ANALYSIS.write_text(json.dumps(s, indent=2))
        print(json.dumps({k: v for k, v in s.items() if k != "cell_gaps"}, indent=1))
