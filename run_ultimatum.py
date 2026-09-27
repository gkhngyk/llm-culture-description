"""Orchestrator for the Ultimatum Game in silico replication.

Commands:
  probe          — single call per model to verify slugs
  smoke          — 1 pair per {arm × offer × model}, full pipeline check
  primary        — 3 models × 2 arms × 5 offers × 20 pairs = 1200 calls
  sweep-profile  — haiku × 3 profile variants × primary design = 1200 calls
  sweep-prompt   — haiku × 3 prompt variants × primary design = 1200 calls
  all            — probe → smoke → primary → sweeps (interactive)

All pair results are appended to results/*.jsonl immediately (resumable).
Raw API traffic is appended to logs/raw_calls.jsonl by ultimatum_sim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from ultimatum_sim import MODELS, PairResult, call_llm, run_pair

PROJECT = Path(__file__).parent
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

OFFERS = [10, 20, 30, 40, 50]
ARMS = ["weird", "smallscale"]
PAIRS_PER_CELL = 20

CONCURRENCY = 8


# --------------------------------------------------------------------------
# Resumable jsonl writer
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_probe() -> None:
    schema = {
        "type": "object",
        "required": ["ack"],
        "properties": {"ack": {"type": "string"}},
    }
    messages = [
        {"role": "system", "content": "You are a probe."},
        {"role": "user", "content": "Reply with a JSON object {\"ack\": \"ok\"}."},
    ]
    for key, slug in MODELS.items():
        try:
            resp = call_llm(slug, messages, schema=schema,
                            session_id=f"probe-{key}", temperature=0.0, max_tokens=40)
            print(f"[probe] {key:7s} {slug:35s} OK -> {resp}")
        except Exception as e:
            print(f"[probe] {key:7s} {slug:35s} FAIL -> {e}", file=sys.stderr)
            sys.exit(2)


def _run_pairs(jobs: list[dict], out_path: Path, tag: str) -> None:
    done_keys = _completed_keys(out_path)
    remaining = [j for j in jobs if _pair_key(j) not in done_keys]
    print(f"[{tag}] total={len(jobs)} done={len(done_keys)} remaining={len(remaining)}")
    if not remaining:
        return

    t0 = time.time()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(run_pair, **j): j for j in remaining}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                res: PairResult = fut.result()
                _append(out_path, res)
                completed += 1
                if completed % 20 == 0 or completed == len(remaining):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (len(remaining) - completed) / max(rate, 1e-6)
                    print(f"[{tag}] {completed}/{len(remaining)} "
                          f"rate={rate:.1f}/s eta={eta:.0f}s failed={failed}")
            except Exception as e:
                failed += 1
                print(f"[{tag}] FAIL {_pair_key(j)}: {e}", file=sys.stderr)
    print(f"[{tag}] finished completed={completed} failed={failed} "
          f"elapsed={time.time()-t0:.1f}s")


def _pair_key(j: dict) -> str:
    return (f"{j['arm']}-off{j['offer']}-{j['model_key']}"
            f"-{j['profile_variant']}-{j['prompt_variant']}-p{j['pair_idx']:03d}")


def _primary_jobs() -> list[dict]:
    jobs = []
    for model_key, slug in MODELS.items():
        for arm in ARMS:
            for offer in OFFERS:
                for i in range(PAIRS_PER_CELL):
                    jobs.append(dict(
                        model_slug=slug, model_key=model_key,
                        arm=arm, offer=offer, pair_idx=i,
                        profile_variant="metric", prompt_variant="formal",
                        temperature=0.4, run_proposer=True,
                        seed_base=10_000 + hash((model_key, arm, offer)) % 100_000,
                    ))
    return jobs


def _sweep_profile_jobs() -> list[dict]:
    jobs = []
    slug = MODELS["haiku"]
    for variant in ["metric", "narrative", "first_person"]:
        if variant == "metric":
            continue  # already covered by primary
        for arm in ARMS:
            for offer in OFFERS:
                for i in range(PAIRS_PER_CELL):
                    jobs.append(dict(
                        model_slug=slug, model_key="haiku",
                        arm=arm, offer=offer, pair_idx=i,
                        profile_variant=variant, prompt_variant="formal",
                        temperature=0.4, run_proposer=True,
                        seed_base=20_000 + hash((variant, arm, offer)) % 100_000,
                    ))
    return jobs


def _sweep_prompt_jobs() -> list[dict]:
    jobs = []
    slug = MODELS["haiku"]
    for variant in ["formal", "conversational", "vignette"]:
        if variant == "formal":
            continue  # already covered by primary
        for arm in ARMS:
            for offer in OFFERS:
                for i in range(PAIRS_PER_CELL):
                    jobs.append(dict(
                        model_slug=slug, model_key="haiku",
                        arm=arm, offer=offer, pair_idx=i,
                        profile_variant="metric", prompt_variant=variant,
                        temperature=0.4, run_proposer=True,
                        seed_base=30_000 + hash((variant, arm, offer)) % 100_000,
                    ))
    return jobs


def cmd_smoke() -> None:
    jobs = []
    for model_key, slug in MODELS.items():
        for arm in ARMS:
            for offer in OFFERS:
                jobs.append(dict(
                    model_slug=slug, model_key=model_key,
                    arm=arm, offer=offer, pair_idx=0,
                    profile_variant="metric", prompt_variant="formal",
                    temperature=0.4, run_proposer=True,
                    seed_base=1_000 + hash((model_key, arm, offer)) % 1000,
                ))
    _run_pairs(jobs, RESULTS / "smoke.jsonl", "smoke")


def cmd_primary() -> None:
    _run_pairs(_primary_jobs(), RESULTS / "primary.jsonl", "primary")


def cmd_sweep_profile() -> None:
    _run_pairs(_sweep_profile_jobs(), RESULTS / "sweep_profile.jsonl", "sweep-profile")


def cmd_sweep_prompt() -> None:
    _run_pairs(_sweep_prompt_jobs(), RESULTS / "sweep_prompt.jsonl", "sweep-prompt")


# --------------------------------------------------------------------------

def main() -> None:
    global CONCURRENCY
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "probe", "smoke", "primary", "sweep-profile", "sweep-prompt",
    ])
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()
    CONCURRENCY = args.concurrency

    {
        "probe": cmd_probe,
        "smoke": cmd_smoke,
        "primary": cmd_primary,
        "sweep-profile": cmd_sweep_profile,
        "sweep-prompt": cmd_sweep_prompt,
    }[args.command]()


if __name__ == "__main__":
    main()
