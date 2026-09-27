"""Turkish figures + numbers for the v2 paper (Study 1 vs numbers test vs Study 2)."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "paper_tr/figs"
OFFERS = [10, 20, 30, 40, 50]
MODELS = [("haiku", "Claude Haiku 4.5"), ("gemma", "Gemma-4-31b-it"), ("gptoss", "GPT-OSS-120b")]
W_COL, S_COL = "#3b6fb6", "#d98b2b"
load = lambda n: [json.loads(l) for l in open(ROOT / "results" / n) if l.strip()]


def rate(rows, **kw):
    v = [r["rejected"] for r in rows if all(r[k] in (x if isinstance(x, tuple) else (x,)) for k, x in kw.items())]
    return float(np.mean(v)) if v else float("nan")


def main():
    out = {}
    S1 = load("primary.jsonl"); S2 = load("nonumber.jsonl"); CN = load("culture_number.jsonl")
    for r in S1:
        r["model_key"] = {"anthropic/claude-haiku-4.5": "haiku", "google/gemma-4-31b-it": "gemma",
                          "openai/gpt-oss-120b": "gptoss"}[r["model_slug"]]

    # ---- Fig: numbers effect (culture-number test), pooled over 3 models and both frames
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    levels = [("hidden", "sayı yok"), ("low", "adalet = 0,3"), ("high", "adalet = 0,6")]
    for ax, (src, title) in zip(axes, [("windfall", "Karşılıksız havuz"), ("earned", "Emekle kazanılmış havuz")]):
        x = np.arange(3); wv = [rate(CN, source=src, numeric=n, arm="weird") for n, _ in levels]
        sv = [rate(CN, source=src, numeric=n, arm="smallscale") for n, _ in levels]
        ax.bar(x - 0.2, wv, 0.4, color=W_COL, label="WEIRD kartı")
        ax.bar(x + 0.2, sv, 0.4, color=S_COL, label="Küçük ölçekli kartı")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in levels]); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.25)
        for n, _ in levels:
            out[f"cn|{src}|{n}|weird"] = round(rate(CN, source=src, numeric=n, arm="weird"), 3)
            out[f"cn|{src}|{n}|smallscale"] = round(rate(CN, source=src, numeric=n, arm="smallscale"), 3)
    axes[0].set_ylabel("reddetme oranı (teklif 10 ve 20)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig15_numbers.png", dpi=170)
    for mk, _ in MODELS:
        for src in ("windfall", "earned"):
            for n, _ in levels:
                for arm in ("weird", "smallscale"):
                    out[f"cn_model|{mk}|{src}|{n}|{arm}"] = round(rate(CN, model_key=mk, source=src, numeric=n, arm=arm), 3)

    # ---- Fig: Study 2 vs Study 1
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), sharey=True)
    panels = [("all", "Üç model birlikte")] + MODELS
    for ax, (mk, title) in zip(axes, panels):
        kw = {} if mk == "all" else {"model_key": mk}
        for arm, col, lab in (("weird", W_COL, "WEIRD"), ("smallscale", S_COL, "Küçük ölçekli")):
            ax.plot(OFFERS, [rate(S2, arm=arm, offer=o, **kw) for o in OFFERS], "o-", color=col,
                    label=f"{lab}, Çalışma 2 (sayısız)")
            ax.plot(OFFERS, [rate(S1, arm=arm, offer=o, **kw) for o in OFFERS], "s:", color=col, alpha=0.6,
                    label=f"{lab}, Çalışma 1 (sayılı)")
            for o in OFFERS:
                out[f"s2|{mk}|{arm}|{o}"] = round(rate(S2, arm=arm, offer=o, **kw), 3)
                out[f"s1|{mk}|{arm}|{o}"] = round(rate(S1, arm=arm, offer=o, **kw), 3)
        ax.set_title(title, fontsize=10); ax.set_xticks(OFFERS); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("teklif (100 üzerinden)"); ax.grid(alpha=0.25)
    axes[0].axhspan(0.40, 0.60, xmax=0.12, color=W_COL, alpha=0.15)
    axes[0].axhspan(0.05, 0.25, xmax=0.12, color=S_COL, alpha=0.15)
    axes[0].set_ylabel("reddetme oranı"); axes[-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGS / "fig16_study2.png", dpi=170)

    # ---- Fig: Study 2 theme coding
    rows = load("nonumber_coding.jsonl")
    agg = defaultdict(lambda: defaultdict(list)); meta = {}
    for r in rows:
        meta[r["tid"]] = r
        for k, v in r["p"].items():
            agg[(r["tid"], r["coder"])][k].append(v)
    P = {k: {t: np.mean(v) for t, v in d.items()} for k, d in agg.items()}
    th = ["kin_community", "survival_need", "game_theory", "fairness_eval", "equal_split_norm", "anonymity", "anti_spite"]
    names = {"kin_community": "Akrabalık /\ntopluluk", "survival_need": "Maddi ihtiyaç", "game_theory": "Oyun teorisi /\nbeklenen değer",
             "fairness_eval": "Adalet\ndeğerlendirmesi", "equal_split_norm": "Eşit paylaşım\nnormu",
             "anonymity": "Anonimlik", "anti_spite": "Reddetme =\nkendine zarar"}
    fig, ax = plt.subplots(figsize=(11.5, 4))
    x = np.arange(len(th))
    for i, (arm, col, lab) in enumerate((("weird", W_COL, "WEIRD"), ("smallscale", S_COL, "Küçük ölçekli"))):
        tids = [t for t in meta if meta[t]["arm"] == arm]
        jv = [np.mean([P[(t, "typesafe/jev-1.13")][h] >= .5 for t in tids]) for h in th]
        sp = [np.mean([P[(t, "respan/span-01")][h] >= .5 for t in tids]) for h in th]
        ax.bar(x + (i - .5) * .38, jv, .38, color=col, label=f"{lab} (Jev)")
        ax.scatter(x + (i - .5) * .38, sp, marker="D", s=22, color="k", zorder=3, label="Span-01" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([names[h] for h in th], fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("temanın görüldüğü metin oranı"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGS / "fig17_themes_s2.png", dpi=170)

    s1_fs = {arm: float(np.median([r["responder_profile"]["fairness_salience"] for r in S1 if r["arm"] == arm]))
             for arm in ("weird", "smallscale")}
    out["s1_fs_median"] = s1_fs
    (ROOT / "results/v2_paper_numbers.json").write_text(json.dumps(out, indent=1))
    for k in sorted(out):
        if k.startswith("cn|") or k.startswith("s2|all") or k.startswith("s1|all"):
            print(k, out[k])
    for mk, _ in MODELS:
        print(mk, "earned", [(n, out[f"cn_model|{mk}|earned|{n}|weird"], out[f"cn_model|{mk}|earned|{n}|smallscale"]) for n in ("hidden", "low", "high")])
        print(mk, "windfall", [(n, out[f"cn_model|{mk}|windfall|{n}|weird"], out[f"cn_model|{mk}|windfall|{n}|smallscale"]) for n in ("hidden", "low", "high")])


if __name__ == "__main__":
    main()
