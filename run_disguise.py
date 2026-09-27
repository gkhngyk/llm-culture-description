"""
Runner + analysis for the disguised-game test (prereg_disguise.md).

  python run_disguise.py preview   # print one control + two disguised prompts, no API
  python run_disguise.py smoke     # 1 profile per model x arm x offer x condition (90 calls)
  python run_disguise.py run       # full design, N=50 (4500 calls), resumable
  python run_disguise.py analyze

Outputs: results/disguise.jsonl, results/disguise_smoke.jsonl,
         results/disguise_analysis.json, plots/disguise_rejection.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

from disguise_test import CONDITIONS, build_disguised_messages, profile_seed, run_one
from ultimatum_sim import MODELS, build_messages, sample_profile

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS = PROJECT / "plots"
OUT = RESULTS / "disguise.jsonl"
SMOKE = RESULTS / "disguise_smoke_v2.jsonl"
ANALYSIS = RESULTS / "disguise_analysis.json"

OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]
N_PAIRS = 50
LOW_OFFERS = (10, 20)
CONCURRENCY = 8


def _jobs(n_pairs: int) -> list[dict]:
    return [dict(model_key=mk, model_slug=slug, arm=arm, offer=off, pair_idx=i, condition=c)
            for mk, slug in MODELS.items() for arm in ARMS for off in OFFERS
            for i in range(n_pairs) for c in CONDITIONS]


def _run(jobs: list[dict], path: Path) -> None:
    done = set()
    if path.exists():
        done = {json.loads(l)["key"] for l in open(path) if l.strip()}
    todo = [j for j in jobs
            if f"{j['condition']}|{j['model_key']}|disg-{j['arm']}-off{j['offer']}-p{j['pair_idx']:03d}" not in done]
    print(f"[disguise] total={len(jobs)} done={len(done)} remaining={len(todo)}")
    t0, ok, fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex, open(path, "a") as f:
        futs = {ex.submit(run_one, **j): j for j in todo}
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n"); f.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {futs[fut]}: {e}", file=sys.stderr)
            if (ok + fail) % 200 == 0:
                print(f"  {ok + fail}/{len(todo)} ok={ok} fail={fail} {time.time() - t0:.0f}s")
    print(f"[disguise] done ok={ok} fail={fail} {time.time() - t0:.0f}s")


def preview() -> None:
    prof = sample_profile("smallscale", "responder", profile_seed("smallscale", 10, 0), "preview-R")
    ctrl = build_messages(prof, "responder", 10, profile_variant="metric", prompt_variant="formal")
    print("=== CONTROL ===\n", ctrl[0]["content"], "\n---\n", ctrl[1]["content"])
    for story in ("wallet", "orchard"):
        m = build_disguised_messages(prof, 10, story)
        print(f"\n=== {story.upper()} ===\n", m[0]["content"], "\n---\n", m[1]["content"])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _mcnemar(a: list[int], b: list[int]) -> dict:
    """Exact McNemar on paired binary outcomes (a=control, b=disguise)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # control accept, disguise reject
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    p = float(stats.binomtest(n01, n, 0.5).pvalue) if n else 1.0
    return {"n_pairs": len(a), "control_rate": round(float(np.mean(a)), 4),
            "disguise_rate": round(float(np.mean(b)), 4),
            "diff": round(float(np.mean(b) - np.mean(a)), 4),
            "discordant_up": n01, "discordant_down": n10, "p": p}


def analyze(path: Path = OUT) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {(r["condition"], r["model_key"], r["profile_id"]): r for r in rows}
    models = sorted({r["model_key"] for r in rows})
    profiles = sorted({r["profile_id"] for r in rows})

    def paired(cond, mks, offers):
        a, b = [], []
        for mk in mks:
            for pid in profiles:
                rc, rd = by.get(("control", mk, pid)), by.get((cond, mk, pid))
                if rc and rd and rc["offer"] in offers:
                    a.append(rc["rejected"]); b.append(rd["rejected"])
        return a, b

    out = {"n_rows": len(rows), "models": models}

    # Primary: low offers, pooled over models, each disguise vs control.
    prim = {}
    for cond in ("wallet", "orchard"):
        t = _mcnemar(*paired(cond, models, LOW_OFFERS))
        t["p_bonf"] = min(1.0, t["p"] * 2)
        prim[cond] = t
    out["primary_low_offer_pooled"] = prim
    ups = [prim[c]["diff"] for c in prim]
    sig = all(prim[c]["p_bonf"] < 0.01 for c in prim)
    if all(u >= 0.10 for u in ups) and sig:
        verdict = "H-D1 supported: recognition mechanism"
    elif all(u < 0.05 for u in ups):
        verdict = "H-D0: recognition mechanism not supported (EV rule)"
    else:
        verdict = "inconclusive / story-dependent"
    out["prereg_verdict"] = verdict

    # Secondary: per model, low offers.
    out["per_model_low_offer"] = {
        f"{mk}|{cond}": _mcnemar(*paired(cond, [mk], LOW_OFFERS))
        for mk in models for cond in ("wallet", "orchard")}

    # Manipulation check: recognition rate per model x condition.
    rec = {}
    for mk in models:
        for cond in CONDITIONS:
            rs = [r["recognition"] for r in rows if r["model_key"] == mk and r["condition"] == cond]
            rec[f"{mk}|{cond}"] = round(float(np.mean(rs)), 4) if rs else None
    out["recognition_rate"] = rec
    valid = {}
    for mk in models:
        c = rec[f"{mk}|control"]
        for cond in ("wallet", "orchard"):
            d = rec[f"{mk}|{cond}"]
            valid[f"{mk}|{cond}"] = bool(d <= 0.01) if c <= 0.03 else bool(d <= c / 3)
    out["manipulation_check_passed"] = valid

    # Cell table: rejection by model x condition x arm x offer.
    cell = defaultdict(list)
    for r in rows:
        cell[(r["model_key"], r["condition"], r["arm"], r["offer"])].append(r["rejected"])
    out["cell_rejection"] = {"|".join(map(str, k)): round(float(np.mean(v)), 4)
                             for k, v in sorted(cell.items())}
    pooled = defaultdict(list)
    for r in rows:
        pooled[(r["condition"], r["offer"])].append(r["rejected"])
    out["pooled_by_offer"] = {f"{c}|{o}": round(float(np.mean(pooled[(c, o)])), 4)
                              for c in CONDITIONS for o in OFFERS if pooled[(c, o)]}
    # Parse health: disguised runs must return take/refuse.
    out["raw_action_counts"] = dict(sorted(
        {f"{c}|{a}": sum(1 for r in rows if r["condition"] == c and r["raw_action"] == a)
         for c in CONDITIONS for a in {r["raw_action"] for r in rows}}.items()))
    return out


def plot(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    models = summary["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    style = {"control": ("k", "o-"), "wallet": ("#3b6fb6", "s--"), "orchard": ("#d98b2b", "^--")}
    for ax, mk in zip(np.atleast_1d(axes), models):
        for cond in CONDITIONS:
            ys = []
            for off in OFFERS:
                v = [summary["cell_rejection"].get(f"{mk}|{cond}|{arm}|{off}") for arm in ARMS]
                v = [x for x in v if x is not None]
                ys.append(np.mean(v) if v else np.nan)
            col, ls = style[cond]
            ax.plot(OFFERS, ys, ls, color=col, label=cond)
        ax.set_title(mk); ax.set_xticks(OFFERS); ax.set_xlabel("offer (of 100)")
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("rejection rate (arms pooled)")
    np.atleast_1d(axes)[-1].legend()
    fig.suptitle("Disguised-game test: rejection by condition")
    fig.tight_layout(); fig.savefig(PLOTS / "disguise_rejection.png", dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preview", "smoke", "run", "analyze"])
    a = ap.parse_args()
    if a.cmd == "preview":
        preview()
    elif a.cmd == "smoke":
        _run(_jobs(1), SMOKE)
        s = analyze(SMOKE)
        print(json.dumps({k: s[k] for k in ("recognition_rate", "pooled_by_offer", "raw_action_counts")}, indent=1))
    else:
        if a.cmd == "run":
            _run(_jobs(N_PAIRS), OUT)
        s = analyze(OUT)
        ANALYSIS.write_text(json.dumps(s, indent=2))
        plot(s)
        print(json.dumps({k: v for k, v in s.items() if k != "cell_rejection"}, indent=1))
