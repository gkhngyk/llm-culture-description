"""English versions of the paper figures (fig13-18) for paper_en/."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "paper_en/figs"
OFFERS = [10, 20, 30, 40, 50]
MODELS = [("haiku", "Claude Haiku 4.5"), ("gemma", "Gemma-4-31b-it"), ("gptoss", "GPT-OSS-120b")]
WC, SC = "#3b6fb6", "#d98b2b"
load = lambda n: [json.loads(l) for l in open(ROOT / "results" / n) if l.strip()]


def rate(rows, **kw):
    v = [r["rejected"] for r in rows if all(r[k] in (x if isinstance(x, tuple) else (x,)) for k, x in kw.items())]
    return float(np.mean(v)) if v else float("nan")


def curves(rows, key, styles, fname):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=True)
    for ax, (mk, name) in zip(axes, MODELS):
        rr = [r for r in rows if r["model_key"] == mk]
        for c, (col, ls, lab) in styles.items():
            ax.plot(OFFERS, [rate(rr, **{key: c, "offer": o}) for o in OFFERS], ls, color=col, label=lab, lw=1.8, ms=5)
        ax.set_title(name, fontsize=10); ax.set_xticks(OFFERS); ax.set_xlabel("offer (out of 100)")
        ax.set_ylim(-0.02, 1.0); ax.grid(alpha=0.25)
    axes[0].set_ylabel("rejection rate"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / fname, dpi=170)


def main():
    D, E = load("disguise.jsonl"), load("entitlement.jsonl")
    curves(D, "condition", {"control": ("k", "o-", "Original game text"), "wallet": (WC, "s--", "Wallet (disguised)"),
                            "orchard": (SC, "^--", "Orchard (disguised)")}, "fig13_disguise.png")
    curves(E, "cell", {"game_windfall": ("k", "o-", "Game, windfall"), "game_earned": ("k", "o--", "Game, earned"),
                       "story_windfall": (SC, "s-", "Story, windfall"), "story_earned": (SC, "s--", "Story, earned")},
           "fig14_entitlement.png")

    CN = load("culture_number.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    levels = [("hidden", "no numbers"), ("low", "fairness = 0.3"), ("high", "fairness = 0.6")]
    for ax, (src, title) in zip(axes, [("windfall", "Windfall pot"), ("earned", "Jointly earned pot")]):
        x = np.arange(3)
        ax.bar(x - .2, [rate(CN, source=src, numeric=n, arm="weird") for n, _ in levels], .4, color=WC, label="WEIRD card")
        ax.bar(x + .2, [rate(CN, source=src, numeric=n, arm="smallscale") for n, _ in levels], .4, color=SC, label="Small-scale card")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in levels]); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("rejection rate (offers 10 and 20)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig15_numbers.png", dpi=170)

    S1, S2 = load("primary.jsonl"), load("nonumber.jsonl")
    slug = {"anthropic/claude-haiku-4.5": "haiku", "google/gemma-4-31b-it": "gemma", "openai/gpt-oss-120b": "gptoss"}
    for r in S1:
        r["model_key"] = slug[r["model_slug"]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), sharey=True)
    for ax, (mk, title) in zip(axes, [("all", "All three models")] + MODELS):
        kw = {} if mk == "all" else {"model_key": mk}
        for arm, col, lab in (("weird", WC, "WEIRD"), ("smallscale", SC, "Small-scale")):
            ax.plot(OFFERS, [rate(S2, arm=arm, offer=o, **kw) for o in OFFERS], "o-", color=col, label=f"{lab}, Study 2 (no numbers)")
            ax.plot(OFFERS, [rate(S1, arm=arm, offer=o, **kw) for o in OFFERS], "s:", color=col, alpha=.6, label=f"{lab}, Study 1 (numbers)")
        ax.set_title(title, fontsize=10); ax.set_xticks(OFFERS); ax.set_ylim(-.02, 1.02); ax.set_xlabel("offer (out of 100)"); ax.grid(alpha=.25)
    axes[0].axhspan(.40, .60, xmax=.12, color=WC, alpha=.15); axes[0].axhspan(.05, .25, xmax=.12, color=SC, alpha=.15)
    axes[0].set_ylabel("rejection rate"); axes[-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGS / "fig16_study2.png", dpi=170)

    rows = load("nonumber_coding.jsonl")
    agg = defaultdict(lambda: defaultdict(list)); meta = {}
    for r in rows:
        meta[r["tid"]] = r
        for k, v in r["p"].items():
            agg[(r["tid"], r["coder"])][k].append(v)
    P = {k: {t: np.mean(v) for t, v in d.items()} for k, d in agg.items()}
    th = ["kin_community", "survival_need", "game_theory", "fairness_eval", "equal_split_norm", "anonymity", "anti_spite"]
    names = {"kin_community": "Kin /\ncommunity", "survival_need": "Material\nneed", "game_theory": "Game theory /\nexpected value",
             "fairness_eval": "Fairness\njudgement", "equal_split_norm": "Equal-split\nnorm", "anonymity": "Anonymity",
             "anti_spite": "Refusal =\nself-harm"}
    fig, ax = plt.subplots(figsize=(11.5, 4)); x = np.arange(len(th))
    for i, (arm, col, lab) in enumerate((("weird", WC, "WEIRD"), ("smallscale", SC, "Small-scale"))):
        tids = [t for t in meta if meta[t]["arm"] == arm]
        ax.bar(x + (i - .5) * .38, [np.mean([P[(t, "typesafe/jev-1.13")][h] >= .5 for t in tids]) for h in th], .38, color=col, label=f"{lab} (Jev)")
        ax.scatter(x + (i - .5) * .38, [np.mean([P[(t, "respan/span-01")][h] >= .5 for t in tids]) for h in th], marker="D", s=22,
                   color="k", zorder=3, label="Span-01" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([names[h] for h in th], fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of texts with theme"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGS / "fig17_themes_s2.png", dpi=170)

    CW = load("card_wording.jsonl")
    words = [("original", "Value-laden\ndescription"), ("minimal", "Minimal\ndescription"), ("ethnographic", "Ethnographic\ndescription")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (mk, t) in zip(axes, [("all", "All three models")] + MODELS):
        x = np.arange(3)
        for i, (arm, col, lab) in enumerate((("weird", WC, "WEIRD"), ("smallscale", SC, "Small-scale"))):
            v = [np.mean([r["rejected"] for r in CW if r["wording"] == w and r["arm"] == arm and r["offer"] == 10
                          and (mk == "all" or r["model_key"] == mk)]) for w, _ in words]
            ax.bar(x + (i - .5) * .38, v, .38, color=col, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in words], fontsize=8); ax.set_title(t, fontsize=10)
        ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=.25)
    axes[0].axhspan(.40, .60, color=WC, alpha=.12); axes[0].axhspan(.05, .25, color=SC, alpha=.12)
    axes[0].set_ylabel("rejection rate (offer 10)"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig18_wording.png", dpi=170)


if __name__ == "__main__":
    main(); print("ok")
