"""Post-hoc blind mechanism coding runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis.blind_coding import (
    run_blind_clustering, sample_reasoning, reveal_and_score,
)
from analysis.stats import load_jsonl
from analysis import plots

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS_DIR = PROJECT / "plots"


def main() -> None:
    primary = load_jsonl(RESULTS / "primary.jsonl")
    samples = sample_reasoning(primary, per_cell=6, seed=42)
    print(f"[blind] sampled {len(samples)} reasoning strings from haiku primary")

    # Coder may choke on a huge prompt — chunk if needed.
    if len(samples) > 90:
        samples = samples[:90]

    result = run_blind_clustering(samples)
    clusters = result.get("clusters", [])
    scored = reveal_and_score(samples, clusters)

    with open(RESULTS / "blind_coding.json", "w") as f:
        json.dump({
            "samples": samples,
            "clusters_raw": clusters,
            "scored": scored,
        }, f, indent=2)

    plots.plot_cluster_report(scored, PLOTS_DIR / "9_reasoning_cluster_dendrogram.png")
    print("[blind] clusters:")
    for c in scored["clusters"]:
        print(f"  - {c['label'][:40]:40s} n={c['n']:2d} "
              f"weird={c['n_weird']:2d} small={c['n_smallscale']:2d} "
              f"purity={c['arm_purity']:.2f}")


if __name__ == "__main__":
    main()
