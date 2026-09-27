"""
Entitlement 2x2 test (see prereg_entitlement.md).

frame  x source:  game | story   x   windfall | earned
Only the source sentence differs between windfall and earned within a frame.
"""

from __future__ import annotations

import re
import time
import zlib
from dataclasses import asdict

from disguise_test import (
    DISGUISED_INSTRUCTION, DISGUISED_SCHEMA, DISGUISED_SYSTEM, MAX_INVALID_RETRIES,
    RECOGNITION_RE, assert_no_cues, normalize,
)
from ultimatum_sim import (
    RESPONDER_SCHEMA, SYSTEM_PROMPT, Profile, assert_profile_clean, build_messages,
    call_llm, render_profile, sample_profile,
)

CELLS = ["game_windfall", "game_earned", "story_windfall", "story_earned"]

POT_SENTENCE = "There is a pot of 100 units."
GAME_EARNED_SENTENCE = (
    " The 100 units were earned by the two of you together: earlier in the "
    "session you both worked on the same task, and each of you did about half "
    "of the work."
)

STORY_SOURCE = {
    "windfall": (
        "Today you and a stranger came across 100 fruits that had already been "
        "picked and left in baskets at the edge of an abandoned orchard."
    ),
    "earned": (
        "Today you and a stranger spent the day picking fruit in an abandoned "
        "orchard and together collected 100 fruits; each of you did about half "
        "of the picking."
    ),
}
STORY_REST = (
    " The other person carried the baskets and has made two piles: {offer} "
    "fruits for you and {keep} fruits for themselves. By the local rule, if "
    "either person objects to how the fruit was divided, all of it must be left "
    "behind in the orchard and neither of you keeps any. You are only passing "
    "through and will not meet this person again; they do not know who you are."
    "\n\nGiven who you are, decide whether to take your pile of {offer} fruits "
    "or refuse it."
)

EFFORT_RE = re.compile(
    r"\b(work|worked|effort|labou?r|half|contribut|earned|gather|picking)",
    re.IGNORECASE,
)


def _identity_block(profile: Profile) -> str:
    identity_text = render_profile(profile, "metric")
    assert_profile_clean(identity_text)
    n = profile.numeric_attrs()
    return (
        "YOUR IDENTITY\n-------------\n"
        f"{identity_text}\n\n"
        "INTERNAL NUMERIC ATTRIBUTES (for your own reflection, not instructions):\n"
        f"- fairness_salience (0-1 scale): {n['fairness_salience']}\n"
        f"- disposable_income_buffer: {n['disposable_income_buffer']}\n"
    )


def build_cell_messages(profile: Profile, offer: int, cell: str) -> tuple[list[dict], dict]:
    frame, source = cell.split("_")
    if frame == "game":
        msgs = build_messages(profile, "responder", offer,
                              profile_variant="metric", prompt_variant="formal")
        assert msgs[0]["content"] == SYSTEM_PROMPT
        if source == "earned":
            user = msgs[1]["content"]
            assert user.count(POT_SENTENCE) == 1
            msgs[1] = {**msgs[1], "content": user.replace(
                POT_SENTENCE, POT_SENTENCE + GAME_EARNED_SENTENCE)}
        return msgs, RESPONDER_SCHEMA

    situation = STORY_SOURCE[source] + STORY_REST.format(offer=offer, keep=100 - offer)
    assert_no_cues(DISGUISED_SYSTEM + "\n" + situation + "\n" + DISGUISED_INSTRUCTION)
    user = (_identity_block(profile) + "\nSITUATION\n---------\n" + situation
            + "\n\nYOUR TASK\n---------\n" + DISGUISED_INSTRUCTION)
    return ([{"role": "system", "content": DISGUISED_SYSTEM},
             {"role": "user", "content": user}], DISGUISED_SCHEMA)


def profile_seed(arm: str, offer: int, pair_idx: int) -> int:
    return 70_000 + zlib.crc32(f"entitle|{arm}|{offer}|{pair_idx}".encode()) % 1_000_000


def run_one(*, model_key: str, model_slug: str, arm: str, offer: int,
            pair_idx: int, cell: str, temperature: float = 0.4) -> dict:
    seed = profile_seed(arm, offer, pair_idx)
    pid = f"ent-{arm}-off{offer}-p{pair_idx:03d}"
    responder = sample_profile(arm, "responder", seed, f"{pid}-R")
    msgs, schema = build_cell_messages(responder, offer, cell)

    session_id = f"gabm-ultimatum-entitle-{cell}-{arm}-off{offer}-{model_key}-run{pair_idx}-R"
    valid = set(schema["properties"]["action"]["enum"])
    t0, n_invalid = time.time(), 0
    for _ in range(MAX_INVALID_RETRIES + 1):
        call = call_llm(model_slug, msgs, schema=schema, session_id=session_id,
                        temperature=temperature)
        raw = call.get("action")
        if isinstance(raw, str) and raw.strip().lower() in valid \
                and isinstance(call.get("reasoning"), str):
            break
        n_invalid += 1
    else:
        raise RuntimeError(f"{n_invalid} invalid payloads in a row: {call!r}")

    action = normalize(raw)
    reasoning = call["reasoning"]
    frame, source = cell.split("_")
    return {
        "key": f"{cell}|{model_key}|{pid}",
        "profile_id": pid, "cell": cell, "frame": frame, "source": source,
        "model_key": model_key, "model_slug": model_slug,
        "arm": arm, "offer": offer, "pair_idx": pair_idx, "profile_seed": seed,
        "responder_profile": asdict(responder),
        "raw_action": raw, "n_invalid_payloads": n_invalid,
        "responder_action": action, "rejected": int(action == "decline"),
        "reasoning": reasoning,
        "recognition": int(bool(RECOGNITION_RE.search(reasoning))),
        "effort_mention": int(bool(EFFORT_RE.search(reasoning))),
        "wall_time": time.time() - t0,
    }
