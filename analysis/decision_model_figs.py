"""Paper figures + extra stats for the decision-model theme coding.

Reads results/decision_model_coding.jsonl (run_decision_model_coding.py) and
writes paper_tr/figs/fig9_themes.png, paper_tr/figs/fig11_ablation.png and
results/decision_model_coding_paper_stats.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
SPAN, JEV = "respan/span-01", "typesafe/jev-1.13"
THEMES_TR = {
    "survival_need": "Maddi ihtiyaç /\nhayatta kalma",
    "kin_community": "Akrabalık /\ntopluluk",
    "game_theory": "Oyun teorisi /\nbeklenen değer",
    "anti_spite": "Reddetme =\nkendine zarar",
    "fairness_eval": "Adalet\ndeğerlendirmesi",
    "equal_split_norm": "Eşit paylaşım\nnormu",
    "anonymity": "Anonimlik /\ntek seferlik",
}


def load():
    rows = [json.loads(l) for l in open(ROOT / "results/decision_model_coding.jsonl")]
    acc = defaultdict(lambda: defaultdict(list)); meta = {}
    for r in rows:
        meta[r["tid"]] = r
        for k, v in r["p"].items():
            acc[(r["tid"], r["coder"])][k].append(v)
    P = {key: {k: float(np.mean(v)) for k, v in d.items()} for key, d in acc.items()}
    return P, meta


def prev(P, meta, corpus, coder, theme, arm):
    v = [P[(t, coder)][theme] for t in meta if meta[t]["corpus"] == corpus and meta[t]["arm"] == arm]
    return float(np.mean(np.array(v) >= 0.5))


def panel(ax, P, meta, corpus, title):
    th = list(THEMES_TR); x = np.arange(len(th)); w = 0.38
    for i, (arm, col, lab) in enumerate([("weird", "#3b6fb6", "WEIRD"), ("smallscale", "#d98b2b", "Küçük ölçekli")]):
        jv = [prev(P, meta, corpus, JEV, t, arm) for t in th]
        sp = [prev(P, meta, corpus, SPAN, t, arm) for t in th]
        xs = x + (i - 0.5) * w
        ax.bar(xs, jv, w, color=col, alpha=0.85, label=f"{lab} (Jev)")
        ax.scatter(xs, sp, marker="D", s=22, color="black", zorder=3,
                   label="Span-01" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([THEMES_TR[t] for t in th], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("temanın görüldüğü metin oranı"); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def main():
    P, meta = load()
    figs = ROOT / "paper_tr/figs"
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    panel(ax, P, meta, "primary", "Tüm birincil yanıtlayıcı gerekçeleri (n = 1500, üç model)")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout(); fig.savefig(figs / "fig9_themes.png", dpi=170)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
    panel(axes[0], P, meta, "blind60", "Haiku örneklemi, orijinal metin (n = 60)")
    panel(axes[1], P, meta, "blind60_red", "Aynı metinler, yüzey işaretleri silinmiş (n = 60)")
    axes[1].legend(fontsize=8, loc="upper right"); fig.tight_layout(); fig.savefig(figs / "fig11_ablation.png", dpi=170)

    s = json.load(open(ROOT / "results/decision_model_coding_analysis.json"))
    out = {"loo_binomial_p_vs_0.5": {}, "kappa_by_corpus": {}}
    n_by = {"blind60": 60, "blind60_red": 60, "primary": 1500}
    for k, a in s["loo_arm_accuracy_from_themes"].items():
        n = n_by[k.split("|")[0]]
        out["loo_binomial_p_vs_0.5"][k] = {"acc": a, "n": n, "hits": round(a * n),
            "p": float(stats.binomtest(round(a * n), n, 0.5, alternative="greater").pvalue)}
    for corpus in n_by:
        tids = [t for t in meta if meta[t]["corpus"] == corpus]
        ks = []
        for th in THEMES_TR:
            a = np.array([P[(t, SPAN)][th] >= .5 for t in tids]); b = np.array([P[(t, JEV)][th] >= .5 for t in tids])
            po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
            ks.append((po - pe) / (1 - pe) if pe < 1 else np.nan)
        out["kappa_by_corpus"][corpus] = {"per_theme": dict(zip(THEMES_TR, np.round(ks, 3).tolist())),
                                          "median": round(float(np.nanmedian(ks)), 3)}
    out["prevalence"] = {f"{c}|{m}|{t}|{a}": prev(P, meta, c, m, t, a)
                         for c in n_by for m in (SPAN, JEV) for t in THEMES_TR for a in ("weird", "smallscale")}
    (ROOT / "results/decision_model_coding_paper_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("loo_binomial_p_vs_0.5", "kappa_by_corpus")}, indent=1))


if __name__ == "__main__":
    main()
