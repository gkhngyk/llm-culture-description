"""Extend the primary design and two variance sweeps from N=20 to N=50.

This runner generates the 30 additional pairs per cell (pair_idx 20..49) for:
  * primary:        3 models × 2 arms × 5 offers × 30 pairs = 900 pairs
  * profile sweep:  Haiku × 2 non-primary variants × 2 arms × 5 offers × 30 = 600 pairs
  * prompt sweep:   Haiku × 2 non-primary variants × 2 arms × 5 offers × 30 = 600 pairs

Total: 2100 new pairs (4200 role calls, ~$7)

Seed formulas and pair_id format exactly match run_ultimatum.py so the new
pairs slot in cleanly next to the existing 0..19 pairs without collision.
Results are appended directly to the existing jsonl files:
  * results/primary.jsonl        (600 → 1500 lines)
  * results/sweep_profile.jsonl  (400 → 1000 lines)
  * results/sweep_prompt.jsonl   (400 → 1000 lines)

Does NOT modify ultimatum_sim.py, run_ultimatum.py, analyze.py, or any
existing results file. Only appends new lines.

Usage:
  python run_extend_to_n50.py count              # dry-run summary
  python run_extend_to_n50.py run --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultimatum_sim import MODELS, PairResult, run_pair

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"

OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]

# Pairs per cell: mevcut N=20'yi N=50'ye çıkarıyoruz
OLD_N = 20
NEW_N = 50
NEW_INDICES = range(OLD_N, NEW_N)  # 20..49 → 30 new pairs per cell

CONCURRENCY = 8

PRIMARY_PATH = RESULTS / "primary.jsonl"
SWEEP_PROFILE_PATH = RESULTS / "sweep_profile.jsonl"
SWEEP_PROMPT_PATH = RESULTS / "sweep_prompt.jsonl"


# ---------------------------------------------------------------------------
# Job generation — exactly mirrors run_ultimatum.py seed/variant patterns
# ---------------------------------------------------------------------------

def primary_jobs() -> list[dict]:
    """Mirror of run_ultimatum._primary_jobs() for pair_idx 20..49."""
    jobs = []
    for model_key, slug in MODELS.items():
        for arm in ARMS:
            for offer in OFFERS:
                for i in NEW_INDICES:
                    jobs.append(dict(
                        model_slug=slug, model_key=model_key,
                        arm=arm, offer=offer, pair_idx=i,
                        profile_variant="metric", prompt_variant="formal",
                        temperature=0.4, run_proposer=True,
                        seed_base=10_000 + hash((model_key, arm, offer)) % 100_000,
                        _out_path=str(PRIMARY_PATH),
                    ))
    return jobs


def sweep_profile_jobs() -> list[dict]:
    """Mirror of run_ultimatum._sweep_profile_jobs() for pair_idx 20..49.

    Only non-primary variants (narrative, first_person); metric is already
    covered by primary.
    """
    jobs = []
    slug = MODELS["haiku"]
    for variant in ["metric", "narrative", "first_person"]:
        if variant == "metric":
            continue
        for arm in ARMS:
            for offer in OFFERS:
                for i in NEW_INDICES:
                    jobs.append(dict(
                        model_slug=slug, model_key="haiku",
                        arm=arm, offer=offer, pair_idx=i,
                        profile_variant=variant, prompt_variant="formal",
                        temperature=0.4, run_proposer=True,
                        seed_base=20_000 + hash((variant, arm, offer)) % 100_000,
                        _out_path=str(SWEEP_PROFILE_PATH),
                    ))
    return jobs


def sweep_prompt_jobs() -> list[dict]:
    """Mirror of run_ultimatum._sweep_prompt_jobs() for pair_idx 20..49."""
    jobs = []
    slug = MODELS["haiku"]
    for variant in ["formal", "conversational", "vignette"]:
        if variant == "formal":
            continue
        for arm in ARMS:
            for offer in OFFERS:
                for i in NEW_INDICES:
                    jobs.append(dict(
                        model_slug=slug, model_key="haiku",
                        arm=arm, offer=offer, pair_idx=i,
                        profile_variant="metric", prompt_variant=variant,
                        temperature=0.4, run_proposer=True,
                        seed_base=30_000 + hash((variant, arm, offer)) % 100_000,
                        _out_path=str(SWEEP_PROMPT_PATH),
                    ))
    return jobs


def all_jobs() -> list[dict]:
    return primary_jobs() + sweep_profile_jobs() + sweep_prompt_jobs()


# ---------------------------------------------------------------------------
# Resume / append helpers (mirror of run_ultimatum)
# ---------------------------------------------------------------------------

def _pair_key(j: dict) -> str:
    """Exact pair_id format used by ultimatum_sim.run_pair."""
    return (f"{j['arm']}-off{j['offer']}-{j['model_key']}"
            f"-{j['profile_variant']}-{j['prompt_variant']}-p{j['pair_idx']:03d}")


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["pair_id"])
            except Exception:
                pass
    return keys


def _append(path: Path, result: PairResult) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------

def _run_one(job: dict) -> tuple[PairResult, str]:
    """Run a single pair. Strip our internal _out_path before passing to run_pair."""
    out_path = job["_out_path"]
    job_clean = {k: v for k, v in job.items() if k != "_out_path"}
    res = run_pair(**job_clean)
    return res, out_path


def cmd_count() -> None:
    pjobs = primary_jobs()
    spjobs = sweep_profile_jobs()
    ptjobs = sweep_prompt_jobs()
    total = len(pjobs) + len(spjobs) + len(ptjobs)

    print(f"[count] primary:       {len(pjobs):4d} new pairs (pair_idx 20..49, 3 models, 2 arms, 5 offers)")
    print(f"[count] sweep_profile: {len(spjobs):4d} new pairs (Haiku, narrative + first_person)")
    print(f"[count] sweep_prompt:  {len(ptjobs):4d} new pairs (Haiku, conversational + vignette)")
    print(f"[count] TOTAL:         {total:4d} new pairs (~{total*2} role calls)")

    # Rough cost estimate: ~$0.0017 per role call blended across 3 models
    est = total * 2 * 0.0017
    print(f"[count] estimated cost: ~${est:.2f}")

    # Check for pre-existing state
    existing = {
        "primary": len(_completed_keys(PRIMARY_PATH)),
        "sweep_profile": len(_completed_keys(SWEEP_PROFILE_PATH)),
        "sweep_prompt": len(_completed_keys(SWEEP_PROMPT_PATH)),
    }
    print(f"[count] existing pair_ids: {existing}")


def cmd_run(concurrency: int) -> None:
    # Collect completed pair_ids across all three target files
    done_by_path: dict[str, set[str]] = {
        str(PRIMARY_PATH): _completed_keys(PRIMARY_PATH),
        str(SWEEP_PROFILE_PATH): _completed_keys(SWEEP_PROFILE_PATH),
        str(SWEEP_PROMPT_PATH): _completed_keys(SWEEP_PROMPT_PATH),
    }

    jobs = all_jobs()
    remaining = [
        j for j in jobs
        if _pair_key(j) not in done_by_path[j["_out_path"]]
    ]
    print(f"[run] total={len(jobs)} done={len(jobs)-len(remaining)} remaining={len(remaining)}")
    if not remaining:
        print("[run] nothing to do")
        return

    t0 = time.time()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_run_one, j): j for j in remaining}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                res, out_path = fut.result()
                _append(Path(out_path), res)
                completed += 1
                if completed % 50 == 0 or completed == len(remaining):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (len(remaining) - completed) / max(rate, 1e-6)
                    print(f"[run] {completed}/{len(remaining)} "
                          f"rate={rate:.1f}/s eta={eta:.0f}s failed={failed}")
            except Exception as e:
                failed += 1
                print(f"[run] FAIL {_pair_key(j)}: {e}", file=sys.stderr)
    print(f"[run] finished completed={completed} failed={failed} "
          f"elapsed={time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["count", "run"])
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    a = p.parse_args()
    {
        "count": lambda: cmd_count(),
        "run":   lambda: cmd_run(a.concurrency),
    }[a.command]()


if __name__ == "__main__":
    main()
