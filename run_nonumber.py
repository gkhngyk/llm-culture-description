"""
Runner + analysis for Study 2 (prereg_nonumber.md).

  python run_nonumber.py preview   # print proposer + responder prompts for one pair, no API
  python run_nonumber.py smoke     # 1 pair per arm x offer x model (60 calls)
  python run_nonumber.py run       # N=50 (3 000 calls), resumable
  python run_nonumber.py analyze

Outputs: results/nonumber.jsonl, results/nonumber_smoke.jsonl,
         results/nonumber_analysis.json, plots/nonumber_vs_study1.png
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

from analysis.stats import chi2_two_sample, cohen_h
from nonumber_test import build_nonumber_messages, profile_seed, run_pair
from ultimatum_sim import MODELS, sample_profile

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS = PROJECT / "plots"
OUT = RESULTS / "nonumber.jsonl"
SMOKE = RESULTS / "nonumber_smoke.jsonl"
ANALYSIS = RESULTS / "nonumber_analysis.json"

OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]
N_PAIRS = 50
CONCURRENCY = 8


def _jobs(n: int) -> list[dict]:
    return [dict(model_key=mk, model_slug=slug, arm=a, offer=o, pair_idx=i)
            for mk, slug in MODELS.items() for a in ARMS for o in OFFERS for i in range(n)]


def _run(jobs: list[dict], path: Path) -> None:
    done = {json.loads(l)["key"] for l in open(path) if l.strip()} if path.exists() else set()
    todo = [j for j in jobs if f"{j['model_key']}|nn-{j['arm']}-off{j['offer']}-p{j['pair_idx']:03d}" not in done]
    print(f"[nonum] total={len(jobs)} done={len(done)} remaining={len(todo)}", flush=True)
    t0, ok, fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex, open(path, "a") as f:
        futs = {ex.submit(run_pair, **j): j for j in todo}
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n"); f.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {futs[fut]}: {e}", file=sys.stderr, flush=True)
            if (ok + fail) % 250 == 0:
                print(f"  {ok + fail}/{len(todo)} ok={ok} fail={fail} {time.time() - t0:.0f}s", flush=True)
    print(f"[nonum] done ok={ok} fail={fail} {time.time() - t0:.0f}s", flush=True)


def preview() -> None:
    for role in ("proposer", "responder"):
        p = sample_profile("weird", role, profile_seed("weird", 10, 0, role[0].upper()), "preview")
        print(f"\n=== {role} ===\n{build_nonumber_messages(p, role, 10)[1]['content']}")


def _rate(rows, **kw):
    v = [r["rejected"] for r in rows if all(r[k] == x for k, x in kw.items())]
    return (float(np.mean(v)) if v else float("nan")), sum(v), len(v)


def analyze(path: Path = OUT) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    models = sorted({r["model_key"] for r in rows})
    out: dict = {"n_pairs": len(rows), "invalid_payloads": sum(r["n_invalid_payloads"] for r in rows)}

    def block(rs):
        w10, xw, nw = _rate(rs, arm="weird", offer=10)
        s10, xs, ns = _rate(rs, arm="smallscale", offer=10)
        chi, p = chi2_two_sample(xw, nw, xs, ns)
        p_bonf = min(1.0, p * 5)
        h = cohen_h(w10, s10)
        res = {"weird_off10": round(w10, 4), "smallscale_off10": round(s10, 4),
               "gap": round(w10 - s10, 4), "p": p, "p_bonf": p_bonf, "cohen_h": round(h, 3),
               "H1": 0.30 <= w10 <= 0.70, "H2": 0.0 <= s10 <= 0.30,
               "H3": (w10 - s10) >= 0.15 and p_bonf < 0.01 and h >= 0.3,
               "H_R": (s10 - w10) >= 0.15 and p_bonf < 0.01 and -h >= 0.3}
        for arm in ARMS:
            curve = [_rate(rs, arm=arm, offer=o)[0] for o in OFFERS]
            rho = stats.spearmanr(OFFERS, curve).statistic if np.std(curve) > 0 else float("nan")
            res[f"curve_{arm}"] = [round(c, 4) for c in curve]
            res[f"H4_{arm}"] = bool(rho < -0.9) if rho == rho else False
            res[f"rho_{arm}"] = None if rho != rho else round(float(rho), 3)
        return res

    out["pooled"] = block(rows)
    out["per_model"] = {mk: block([r for r in rows if r["model_key"] == mk]) for mk in models}

    devs = []
    for arm in ARMS:
        for o in OFFERS:
            rates = [_rate([r for r in rows if r["model_key"] == mk], arm=arm, offer=o)[0] for mk in models]
            devs.append(max(abs(x - np.mean(rates)) for x in rates))
    out["H5_max_dev"] = round(float(max(devs)), 4)
    out["H5"] = out["H5_max_dev"] <= 0.15

    pl = out["pooled"]
    out["finding_H1_H2_H3"] = bool(pl["H1"] and pl["H2"] and pl["H3"])

    # H-S1: Study 2 vs Study 1 at offer 10, pooled.
    s1 = [json.loads(l) for l in open(RESULTS / "primary.jsonl")]
    a = [r["rejected"] for r in s1 if r["offer"] == 10]
    b = [r["rejected"] for r in rows if r["offer"] == 10]
    p = float(stats.fisher_exact([[sum(b), len(b) - sum(b)], [sum(a), len(a) - sum(a)]])[1])
    out["H_S1"] = {"study1_off10": round(float(np.mean(a)), 4), "study2_off10": round(float(np.mean(b)), 4),
                   "diff": round(float(np.mean(b) - np.mean(a)), 4), "p": p,
                   "supported": bool(np.mean(b) - np.mean(a) >= 0.10 and p < 0.01)}
    d = [json.loads(l) for l in open(RESULTS / "disguise.jsonl")]
    out["H_S1"]["drift_check_disguise_control_off10"] = round(float(np.mean(
        [r["rejected"] for r in d if r["condition"] == "control" and r["offer"] == 10])), 4)

    out["proposer_predicted_reject"] = {
        f"{arm}|{o}": round(float(np.mean([r["proposer_expected"] == "decline" for r in rows
                                           if r["arm"] == arm and r["offer"] == o])), 4)
        for arm in ARMS for o in OFFERS}
    return out


def plot(s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    s1 = [json.loads(l) for l in open(RESULTS / "primary.jsonl")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, arm in zip(axes, ARMS):
        ax.plot(OFFERS, s["pooled"][f"curve_{arm}"], "o-", color="#d98b2b", label="Study 2 (no numbers)")
        ax.plot(OFFERS, [_rate(s1, arm=arm, offer=o)[0] for o in OFFERS], "s--", color="k",
                label="Study 1 (with numbers)")
        band = (0.40, 0.60) if arm == "weird" else (0.05, 0.25)
        ax.axhspan(*band, xmin=0, xmax=0.25, color="grey", alpha=0.2, label="Henrich human band (<20%)")
        ax.set_title(arm); ax.set_xticks(OFFERS); ax.set_ylim(-0.02, 1); ax.grid(alpha=0.25)
        ax.set_xlabel("offer (of 100)")
    axes[0].set_ylabel("rejection rate (3 models pooled)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "nonumber_vs_study1.png", dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preview", "smoke", "run", "analyze"])
    a = ap.parse_args()
    if a.cmd == "preview":
        preview()
    elif a.cmd == "smoke":
        _run(_jobs(1), SMOKE)
        rows = [json.loads(l) for l in open(SMOKE)]
        print("invalid payloads:", sum(r["n_invalid_payloads"] for r in rows), "rows:", len(rows))
    else:
        if a.cmd == "run":
            _run(_jobs(N_PAIRS), OUT)
        s = analyze(OUT)
        ANALYSIS.write_text(json.dumps(s, indent=2))
        plot(s)
        print(json.dumps(s, indent=1))
