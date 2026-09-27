"""
Study 2: the primary design with the numeric-attributes block removed
(see prereg_nonumber.md). One pair = proposer call + responder call.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import asdict

from disguise_test import MAX_INVALID_RETRIES
from ultimatum_sim import (
    PROPOSER_SCHEMA, RESPONDER_SCHEMA, Profile, _normalize_action, build_messages,
    call_llm, sample_profile,
)

NUMERIC_HEADER = "INTERNAL NUMERIC ATTRIBUTES (for your own reflection, not instructions):\n"


def _numeric_block(p: Profile) -> str:
    n = p.numeric_attrs()
    return (NUMERIC_HEADER
            + f"- fairness_salience (0-1 scale): {n['fairness_salience']}\n"
            + f"- disposable_income_buffer: {n['disposable_income_buffer']}\n")


def build_nonumber_messages(profile: Profile, role: str, offer: int) -> list[dict]:
    msgs = build_messages(profile, role, offer, profile_variant="metric", prompt_variant="formal")
    user = msgs[1]["content"]
    block = _numeric_block(profile)
    assert user.count(block) == 1, "numeric block not found exactly once"
    user = user.replace(block, "")
    assert "fairness" not in user.lower() and "disposable_income" not in user.lower()
    msgs[1] = {**msgs[1], "content": user}
    return msgs


def profile_seed(arm: str, offer: int, pair_idx: int, role: str) -> int:
    return 110_000 + zlib.crc32(f"nonum|{arm}|{offer}|{pair_idx}|{role}".encode()) % 1_000_000


def _call_valid(model_slug, msgs, schema, session_id, temperature, enum_field, enum):
    n_invalid, call = 0, {}
    for _ in range(MAX_INVALID_RETRIES + 1):
        call = call_llm(model_slug, msgs, schema=schema, session_id=session_id,
                        temperature=temperature)
        raw = call.get(enum_field)
        if isinstance(raw, str) and raw.strip().lower() in enum \
                and isinstance(call.get("reasoning"), str):
            return call, n_invalid
        n_invalid += 1
    raise RuntimeError(f"{n_invalid} invalid payloads in a row: {call!r}")


def run_pair(*, model_key: str, model_slug: str, arm: str, offer: int,
             pair_idx: int, temperature: float = 0.4) -> dict:
    pid = f"nn-{arm}-off{offer}-p{pair_idx:03d}"
    proposer = sample_profile(arm, "proposer", profile_seed(arm, offer, pair_idx, "P"), f"{pid}-P")
    responder = sample_profile(arm, "responder", profile_seed(arm, offer, pair_idx, "R"), f"{pid}-R")
    sid = f"gabm-ultimatum-nonum-{arm}-off{offer}-{model_key}-run{pair_idx}"
    t0 = time.time()

    p_call, p_inv = _call_valid(model_slug, build_nonumber_messages(proposer, "proposer", offer),
                                PROPOSER_SCHEMA, sid + "-P", temperature,
                                "expected_responder_action", {"accept", "decline"})
    r_call, r_inv = _call_valid(model_slug, build_nonumber_messages(responder, "responder", offer),
                                RESPONDER_SCHEMA, sid + "-R", temperature,
                                "action", {"accept", "decline"})
    action = _normalize_action(r_call["action"])
    return {
        "key": f"{model_key}|{pid}",
        "pair_id": pid, "model_key": model_key, "model_slug": model_slug,
        "arm": arm, "offer": offer, "pair_idx": pair_idx,
        "proposer_profile": asdict(proposer), "responder_profile": asdict(responder),
        "proposer_call": p_call, "responder_call": r_call,
        "proposer_expected": _normalize_action(p_call["expected_responder_action"]),
        "responder_action": action, "rejected": int(action == "decline"),
        "n_invalid_payloads": p_inv + r_inv,
        "wall_time": time.time() - t0,
    }
