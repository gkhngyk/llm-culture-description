"""
Runner + analysis for the entitlement 2x2 test (prereg_entitlement.md).

  python run_entitlement.py preview   # print the four prompts for one profile, no API
  python run_entitlement.py smoke     # 1 profile per model x arm x offer x cell (120 calls)
  python run_entitlement.py run       # N=50 (6000 calls), resumable
  python run_entitlement.py analyze

Outputs: results/entitlement.jsonl, results/entitlement_smoke.jsonl,
         results/entitlement_analysis.json, plots/entitlement_2x2.png
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

from entitlement_test import CELLS, build_cell_messages, profile_seed, run_one
from run_disguise import _mcnemar
from ultimatum_sim import MODELS, sample_profile

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS = PROJECT / "plots"
OUT = RESULTS / "entitlement.jsonl"
SMOKE = RESULTS / "entitlement_smoke.jsonl"
ANALYSIS = RESULTS / "entitlement_analysis.json"

OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]
N_PAIRS = 50
LOW_OFFERS = (10, 20)
CONCURRENCY = 8


def _jobs(n_pairs: int) -> list[dict]:
    return [dict(model_key=mk, model_slug=slug, arm=arm, offer=off, pair_idx=i, cell=c)
            for mk, slug in MODELS.items() for arm in ARMS for off in OFFERS
            for i in range(n_pairs) for c in CELLS]


def _key(j: dict) -> str:
    return f"{j['cell']}|{j['model_key']}|ent-{j['arm']}-off{j['offer']}-p{j['pair_idx']:03d}"


def _run(jobs: list[dict], path: Path) -> None:
    done = {json.loads(l)["key"] for l in open(path) if l.strip()} if path.exists() else set()
    todo = [j for j in jobs if _key(j) not in done]
    print(f"[entitle] total={len(jobs)} done={len(done)} remaining={len(todo)}")
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
            if (ok + fail) % 250 == 0:
                print(f"  {ok + fail}/{len(todo)} ok={ok} fail={fail} {time.time() - t0:.0f}s")
    print(f"[entitle] done ok={ok} fail={fail} {time.time() - t0:.0f}s")


def preview() -> None:
    prof = sample_profile("weird", "responder", profile_seed("weird", 10, 0), "preview-R")
    for cell in CELLS:
        msgs, _ = build_cell_messages(prof, 10, cell)
        situation = msgs[1]["content"].split("SITUATION\n---------\n")[1].split("\n\nYOUR TASK")[0]
        print(f"\n=== {cell} ===\n[system] {msgs[0]['content']}\n[situation] {situation}")


def analyze(path: Path = OUT) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {(r["cell"], r["model_key"], r["profile_id"]): r for r in rows}
    models = sorted({r["model_key"] for r in rows})
    profiles = sorted({r["profile_id"] for r in rows})

    def paired(c_a, c_b, mks, offers=LOW_OFFERS):
        a, b = [], []
        for mk in mks:
            for pid in profiles:
                ra, rb = by.get((c_a, mk, pid)), by.get((c_b, mk, pid))
                if ra and rb and ra["offer"] in offers:
                    a.append(ra["rejected"]); b.append(rb["rejected"])
        return a, b

    out = {"n_rows": len(rows), "models": models,
           "invalid_payloads": sum(r["n_invalid_payloads"] for r in rows)}

    prim = {"C1_game": _mcnemar(*paired("game_windfall", "game_earned", models)),
            "C2_story": _mcnemar(*paired("story_windfall", "story_earned", models))}
    for t in prim.values():
        t["p_bonf"] = min(1.0, t["p"] * 2)
    out["primary"] = prim
    rises = [prim[k]["diff"] for k in prim]
    if all(d >= 0.05 for d in rises) and all(prim[k]["p_bonf"] < 0.01 for k in prim):
        verdict = "H-E1 supported: entitlement effect"
    elif all(d < 0.02 for d in rises):
        verdict = "H-E0: entitlement effect not supported"
    else:
        verdict = "frame-dependent / inconclusive"
    out["prereg_verdict"] = verdict

    out["per_model"] = {
        f"{mk}|{name}": _mcnemar(*paired(a, b, [mk]))
        for mk in models
        for name, a, b in [("C1_game", "game_windfall", "game_earned"),
                           ("C2_story", "story_windfall", "story_earned"),
                           ("frame_windfall", "game_windfall", "story_windfall")]}
    out["frame_contrast_windfall"] = _mcnemar(*paired("game_windfall", "story_windfall", models))

    low = {c: np.mean([r["rejected"] for r in rows if r["cell"] == c and r["offer"] in LOW_OFFERS])
           for c in CELLS}
    out["low_offer_rate"] = {c: round(float(v), 4) for c, v in low.items()}
    out["interaction"] = round(float((low["story_earned"] - low["story_windfall"])
                                     - (low["game_earned"] - low["game_windfall"])), 4)

    manip = {}
    for mk in models:
        for frame in ("game", "story"):
            e = {s: np.mean([r["effort_mention"] for r in rows
                             if r["model_key"] == mk and r["cell"] == f"{frame}_{s}"])
                 for s in ("windfall", "earned")}
            manip[f"{mk}|{frame}"] = {"windfall": round(float(e["windfall"]), 3),
                                      "earned": round(float(e["earned"]), 3),
                                      "noticed": bool(e["earned"] - e["windfall"] >= 0.20)}
    out["manipulation_check_effort"] = manip
    out["recognition_rate"] = {
        f"{mk}|{c}": round(float(np.mean([r["recognition"] for r in rows
                                          if r["model_key"] == mk and r["cell"] == c])), 4)
        for mk in models for c in CELLS}

    cell = defaultdict(list)
    for r in rows:
        cell[(r["model_key"], r["cell"], r["arm"], r["offer"])].append(r["rejected"])
    out["cell_rejection"] = {"|".join(map(str, k)): round(float(np.mean(v)), 4)
                             for k, v in sorted(cell.items())}
    return out


def plot(s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    models = s["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 3.9), sharey=True)
    style = {"game_windfall": ("k", "o-"), "game_earned": ("k", "o--"),
             "story_windfall": ("#d98b2b", "s-"), "story_earned": ("#d98b2b", "s--")}
    for ax, mk in zip(np.atleast_1d(axes), models):
        for c in CELLS:
            ys = [np.nanmean([s["cell_rejection"].get(f"{mk}|{c}|{a}|{o}", np.nan) for a in ARMS])
                  for o in OFFERS]
            col, ls = style[c]
            ax.plot(OFFERS, ys, ls, color=col, label=c)
        ax.set_title(mk); ax.set_xticks(OFFERS); ax.set_xlabel("offer (of 100)"); ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("rejection rate (arms pooled)")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("Entitlement 2×2: frame × source of the 100")
    fig.tight_layout(); fig.savefig(PLOTS / "entitlement_2x2.png", dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preview", "smoke", "run", "analyze"])
    a = ap.parse_args()
    if a.cmd == "preview":
        preview()
    elif a.cmd == "smoke":
        _run(_jobs(1), SMOKE)
        s = analyze(SMOKE)
        print(json.dumps({k: s[k] for k in ("low_offer_rate", "manipulation_check_effort",
                                            "invalid_payloads")}, indent=1))
    else:
        if a.cmd == "run":
            _run(_jobs(N_PAIRS), OUT)
        s = analyze(OUT)
        ANALYSIS.write_text(json.dumps(s, indent=2))
        plot(s)
        print(json.dumps({k: v for k, v in s.items() if k != "cell_rejection"}, indent=1))
