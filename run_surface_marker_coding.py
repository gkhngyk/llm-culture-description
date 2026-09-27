"""Runner for the surface-marker blind coding follow-up analysis.

Loads the existing results/blind_coding.json, applies regex redaction of
culturally-loaded surface tokens, ships the redacted texts to Gemma for
fresh clustering, and compares purities against the original run.

Usage:
  python run_surface_marker_coding.py preview   # show redactions, no API
  python run_surface_marker_coding.py run       # ~1 API call to Gemma

Does NOT touch results/blind_coding.json or any other existing file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis.blind_coding import reveal_and_score
from analysis.surface_marker_coding import (
    strip_markers, run_surface_marker_coding,
    compare_purities, plot_surface_comparison,
)

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
PLOTS = PROJECT / "plots"

ORIGINAL_PATH = RESULTS / "blind_coding.json"
OUT_PATH = RESULTS / "surface_marker_coding.json"
PLOT_PATH = PLOTS / "11_surface_marker_coding.png"


def _load_original() -> dict:
    if not ORIGINAL_PATH.exists():
        print(f"[smc] cannot find {ORIGINAL_PATH}; "
              f"run run_blind_coding.py first", file=sys.stderr)
        sys.exit(1)
    with open(ORIGINAL_PATH) as f:
        return json.load(f)


def cmd_preview() -> None:
    data = _load_original()
    samples = data["samples"]
    total_markers = 0
    for i, s in enumerate(samples[:6]):
        red, matches = strip_markers(s["text"])
        total_markers += len(matches)
        print(f"--- sample {i} (hidden arm={s['hidden_arm']}, "
              f"offer={s['hidden_offer']}) ---")
        print(f"ORIGINAL:  {s['text'][:220]}")
        print(f"REDACTED:  {red[:220]}")
        print(f"matches ({len(matches)}): {matches[:8]}")
        print()
    # Summary across full set
    total = 0
    for s in samples:
        _, m = strip_markers(s["text"])
        total += len(m)
    print(f"[preview] total surface markers matched across "
          f"{len(samples)} samples: {total}")
    print(f"[preview] avg markers per sample: {total/len(samples):.2f}")


def cmd_run() -> None:
    data = _load_original()
    samples = data["samples"]
    original_scored = data.get("scored", {"clusters": data.get("clusters_raw", [])})
    # If the stored "scored" is empty, recompute it from the raw clusters
    if not original_scored.get("clusters"):
        original_scored = reveal_and_score(samples, data.get("clusters_raw", []))

    print(f"[smc] {len(samples)} samples loaded from {ORIGINAL_PATH}")
    print(f"[smc] original mean cluster count: {len(original_scored.get('clusters', []))}")

    print("[smc] running surface-marker-stripped blind clustering…")
    result = run_surface_marker_coding(samples)
    print(f"[smc] total markers removed: {result['total_markers_removed']}")

    # Reveal and score the new clusters
    redacted_scored = reveal_and_score(
        result["redacted_samples"],
        result["clusters_raw"],
    )
    print("[smc] redacted cluster purities:")
    for c in redacted_scored["clusters"]:
        print(f"  - {c['label'][:40]:40s} n={c['n']:2d} "
              f"weird={c['n_weird']:2d} small={c['n_smallscale']:2d} "
              f"purity={c['arm_purity']:.2f}")

    comparison = compare_purities(original_scored, redacted_scored)
    print("\n[smc] PURITY COMPARISON:")
    print(f"  original mean purity:  {comparison['original_mean_purity']:.3f}")
    print(f"  redacted mean purity:  {comparison['redacted_mean_purity']:.3f}")
    print(f"  delta:                 {comparison['delta']:+.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "redacted_samples": result["redacted_samples"],
            "total_markers_removed": result["total_markers_removed"],
            "clusters_raw": result["clusters_raw"],
            "scored": redacted_scored,
            "original_scored": original_scored,
            "comparison": comparison,
        }, f, indent=2)
    print(f"[smc] wrote {OUT_PATH}")

    plot_surface_comparison(
        original_scored, redacted_scored, comparison, PLOT_PATH,
    )
    print(f"[smc] plot: {PLOT_PATH}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["preview", "run"])
    a = p.parse_args()
    {
        "preview": cmd_preview,
        "run":     cmd_run,
    }[a.command]()


if __name__ == "__main__":
    main()
