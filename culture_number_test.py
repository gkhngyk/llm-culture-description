"""
Culture-or-number test (see prereg_culture_number.md).

Cells: numeric block (hidden / low / high) x source (windfall / earned / placebo)
x frame (game / story). Identity text per arm is unchanged; the numeric block
is either removed or fixed to the same values for both arms.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import asdict

from disguise_test import (
    DISGUISED_INSTRUCTION, DISGUISED_SCHEMA, DISGUISED_SYSTEM, MAX_INVALID_RETRIES,
    RECOGNITION_RE, assert_no_cues, normalize,
)
from entitlement_test import (
    EFFORT_RE, GAME_EARNED_SENTENCE, POT_SENTENCE, STORY_REST, STORY_SOURCE,
)
from ultimatum_sim import (
    RESPONDER_SCHEMA, SYSTEM_PROMPT, Profile, assert_profile_clean, build_messages,
    call_llm, render_profile, sample_profile,
)

NUMERIC = {"hidden": None, "low": (0.3, 300.0), "high": (0.6, 300.0)}
SOURCES = ["windfall", "earned", "placebo"]
FRAMES = ["game", "story"]
CELLS = [f"{f}|{s}|{n}" for f in FRAMES for s in SOURCES for n in NUMERIC]

GAME_PLACEBO_SENTENCE = (
    " Earlier in the session the two of you did a short practice task together, "
    "each doing about half of the work; that practice task was unpaid and had "
    "nothing to do with this money. The 100 units are a separate gift from the "
    "experimenters, not something either of you earned."
)
STORY_PLACEBO_SOURCE = (
    "Yesterday you and this stranger volunteered together clearing brush along "
    "a road, each doing about half of the work, without pay; that work had "
    "nothing to do with any fruit. Today the two of you came across 100 fruits "
    "that had already been picked and left in baskets at the edge of an "
    "abandoned orchard; neither of you picked them."
)

NUMERIC_HEADER = "INTERNAL NUMERIC ATTRIBUTES (for your own reflection, not instructions):\n"


def _numeric_block(fs: float, buf: float) -> str:
    return (NUMERIC_HEADER
            + f"- fairness_salience (0-1 scale): {round(fs, 3)}\n"
            + f"- disposable_income_buffer: {round(buf, 1)}\n")


def _apply_numeric(user: str, profile: Profile, numeric: str) -> str:
    """Replace the profile's own numeric block with the cell's version."""
    own = _numeric_block(profile.fairness_salience, profile.disposable_income_buffer)
    assert user.count(own) == 1, "numeric block not found exactly once"
    if NUMERIC[numeric] is None:
        return user.replace(own, "")
    return user.replace(own, _numeric_block(*NUMERIC[numeric]))


def build_cell(profile: Profile, offer: int, cell: str) -> tuple[list[dict], dict]:
    frame, source, numeric = cell.split("|")
    if frame == "game":
        msgs = build_messages(profile, "responder", offer,
                              profile_variant="metric", prompt_variant="formal")
        assert msgs[0]["content"] == SYSTEM_PROMPT
        user = msgs[1]["content"]
        assert user.count(POT_SENTENCE) == 1
        extra = {"windfall": "", "earned": GAME_EARNED_SENTENCE,
                 "placebo": GAME_PLACEBO_SENTENCE}[source]
        user = user.replace(POT_SENTENCE, POT_SENTENCE + extra)
        msgs[1] = {**msgs[1], "content": _apply_numeric(user, profile, numeric)}
        return msgs, RESPONDER_SCHEMA

    src = STORY_PLACEBO_SOURCE if source == "placebo" else STORY_SOURCE[source]
    situation = src + STORY_REST.format(offer=offer, keep=100 - offer)
    assert_no_cues(DISGUISED_SYSTEM + "\n" + situation + "\n" + DISGUISED_INSTRUCTION)
    identity_text = render_profile(profile, "metric")
    assert_profile_clean(identity_text)
    user = ("YOUR IDENTITY\n-------------\n" + identity_text + "\n\n"
            + _numeric_block(profile.fairness_salience, profile.disposable_income_buffer)
            + "\nSITUATION\n---------\n" + situation
            + "\n\nYOUR TASK\n---------\n" + DISGUISED_INSTRUCTION)
    user = _apply_numeric(user, profile, numeric)
    return ([{"role": "system", "content": DISGUISED_SYSTEM},
             {"role": "user", "content": user}], DISGUISED_SCHEMA)


def profile_seed(arm: str, offer: int, pair_idx: int) -> int:
    return 90_000 + zlib.crc32(f"culnum|{arm}|{offer}|{pair_idx}".encode()) % 1_000_000


def run_one(*, model_key: str, model_slug: str, arm: str, offer: int,
            pair_idx: int, cell: str, temperature: float = 0.4) -> dict:
    seed = profile_seed(arm, offer, pair_idx)
    pid = f"cn-{arm}-off{offer}-p{pair_idx:03d}"
    responder = sample_profile(arm, "responder", seed, f"{pid}-R")
    msgs, schema = build_cell(responder, offer, cell)
    frame, source, numeric = cell.split("|")

    session_id = (f"gabm-ultimatum-culnum-{frame}-{source}-{numeric}-{arm}-off{offer}"
                  f"-{model_key}-run{pair_idx}-R")
    valid = set(schema["properties"]["action"]["enum"])
    t0, n_invalid, call, raw = time.time(), 0, {}, None
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
    return {
        "key": f"{cell}|{model_key}|{pid}",
        "profile_id": pid, "cell": cell, "frame": frame, "source": source,
        "numeric": numeric, "model_key": model_key, "model_slug": model_slug,
        "arm": arm, "offer": offer, "pair_idx": pair_idx, "profile_seed": seed,
        "responder_profile": asdict(responder),
        "raw_action": raw, "n_invalid_payloads": n_invalid,
        "responder_action": action, "rejected": int(action == "decline"),
        "reasoning": reasoning,
        "recognition": int(bool(RECOGNITION_RE.search(reasoning))),
        "effort_mention": int(bool(EFFORT_RE.search(reasoning))),
        "wall_time": time.time() - t0,
    }
