"""Turkish paper figures + table numbers for follow-ups D (disguise) and E (entitlement)."""
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
MODELS = [("haiku", "Claude Haiku 4.5"), ("gemma", "Gemma-4-31b-it"), ("gptoss", "GPT-OSS-120b")]
OFFERS = [10, 20, 30, 40, 50]


def load(name):
    return [json.loads(l) for l in open(ROOT / "results" / name) if l.strip()]


def rate(rows, **kw):
    v = [r["rejected"] for r in rows if all(r[k] in (x if isinstance(x, tuple) else (x,)) for k, x in kw.items())]
    return float(np.mean(v)), len(v)


def curves(ax, rows, key, conds, styles, title):
    for c in conds:
        ys = [rate(rows, **{key: c, "offer": o})[0] for o in OFFERS]
        col, ls, lab = styles[c]
        ax.plot(OFFERS, ys, ls, color=col, label=lab, lw=1.8, ms=5)
    ax.set_title(title, fontsize=10); ax.set_xticks(OFFERS); ax.set_xlabel("teklif (100 üzerinden)")
    ax.set_ylim(-0.02, 1.0); ax.grid(alpha=0.25)


def main():
    D, E = load("disguise.jsonl"), load("entitlement.jsonl")
    out = {"D": {}, "E": {}}
    # --- D table
    for mk, _ in MODELS:
        for c in ("control", "wallet", "orchard"):
            rr = [r for r in D if r["model_key"] == mk and r["condition"] == c]
            out["D"][f"{mk}|{c}"] = {
                "low_rej": round(rate(rr, offer=(10, 20))[0], 3),
                "recog": round(float(np.mean([r["recognition"] for r in rr])), 3), "n": len(rr)}
    for c in ("control", "wallet", "orchard"):
        out["D"][f"all|{c}"] = {"low_rej": round(rate(D, condition=c, offer=(10, 20))[0], 3)}
    # --- E table
    cells = ["game_windfall", "game_earned", "story_windfall", "story_earned"]
    for mk, _ in MODELS:
        for c in cells:
            rr = [r for r in E if r["model_key"] == mk and r["cell"] == c]
            out["E"][f"{mk}|{c}"] = {"low_rej": round(rate(rr, offer=(10, 20))[0], 3),
                                     "off10": round(rate(rr, offer=10)[0], 3),
                                     "effort": round(float(np.mean([r["effort_mention"] for r in rr])), 3)}
    for c in cells:
        out["E"][f"all|{c}"] = {"low_rej": round(rate(E, cell=c, offer=(10, 20))[0], 3)}
    (ROOT / "results/followup_DE_paper_numbers.json").write_text(json.dumps(out, indent=1))

    # --- Fig D
    st = {"control": ("k", "o-", "Orijinal oyun metni"), "wallet": ("#3b6fb6", "s--", "Cüzdan (gizli)"),
          "orchard": ("#d98b2b", "^--", "Meyve bahçesi (gizli)")}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=True)
    for ax, (mk, name) in zip(axes, MODELS):
        curves(ax, [r for r in D if r["model_key"] == mk], "condition", list(st), st, name)
    axes[0].set_ylabel("reddetme oranı"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig13_disguise.png", dpi=170)
    # --- Fig E
    st = {"game_windfall": ("k", "o-", "Oyun, karşılıksız"), "game_earned": ("k", "o--", "Oyun, kazanılmış"),
          "story_windfall": ("#d98b2b", "s-", "Hikâye, karşılıksız"),
          "story_earned": ("#d98b2b", "s--", "Hikâye, kazanılmış")}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=True)
    for ax, (mk, name) in zip(axes, MODELS):
        curves(ax, [r for r in E if r["model_key"] == mk], "cell", list(st), st, name)
    axes[0].set_ylabel("reddetme oranı"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig14_entitlement.png", dpi=170)
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
