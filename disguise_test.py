"""
Disguised-game test (see prereg_disguise.md).

Same responder profiles, same payoffs; the Ultimatum framing is either kept
(control, original prompts) or replaced by a cover story with no experimental
or game vocabulary (wallet, orchard).
"""

from __future__ import annotations

import re
import time
import zlib
from dataclasses import asdict

from ultimatum_sim import (
    RESPONDER_SCHEMA, SYSTEM_PROMPT, Profile, _normalize_action,
    assert_profile_clean, build_messages, call_llm, render_profile, sample_profile,
)

CONDITIONS = ["control", "wallet", "orchard"]
MAX_INVALID_RETRIES = 3

DISGUISED_SYSTEM = (
    "You are the person described below. Decide what to do as that person "
    "would, based on who you are and your situation."
)

STORIES = {
    "wallet": (
        "Earlier today you and a stranger found the same lost wallet at the same "
        "moment and returned it to its owner together. The owner left a reward of "
        "100 coins with the other person and asked them to divide it between the "
        "two of you. They are handing you {offer} coins and keeping {keep} coins "
        "for themselves. The owner's note says the reward is valid only if you "
        "take your share; if you refuse it, the whole reward is cancelled and "
        "neither of you keeps anything. "
        "You will never see this person again, and they do not know your name.\n\n"
        "Given who you are, decide whether to take the {offer} coins or refuse them."
    ),
    "orchard": (
        "Today you and a stranger spent the day gathering fruit from an abandoned "
        "orchard and together collected 100 fruits. The other person carried the "
        "baskets and has made two piles: {offer} fruits for you and {keep} fruits "
        "for themselves. By the local rule, if either person objects to how the "
        "fruit was divided, all of it must be left behind in the orchard and "
        "neither of you keeps any. You are only passing through and will not meet this person "
        "again; they do not know who you are.\n\n"
        "Given who you are, decide whether to take your pile of {offer} fruits or "
        "refuse it."
    ),
}

DISGUISED_INSTRUCTION = (
    "Decide whether to take or refuse. "
    "Provide two to three sentences of reasoning for your decision."
)

DISGUISED_SCHEMA = {
    "type": "object",
    "required": ["action", "reasoning"],
    "properties": {
        "action": {"type": "string", "enum": ["take", "refuse"]},
        "reasoning": {"type": "string"},
    },
}

# Vocabulary that cues the experimental paradigm; must not appear in any
# disguised prompt (system + user).
CUE_TERMS = [
    r"\bexperiment", r"\bparticipant", r"\boffer", r"\bpropos", r"\bsplit",
    r"\bpot\b", r"\bunits?\b", r"\baccept", r"\bdecline", r"one-shot",
    r"\banonym", r"\bround\b", r"\bgame\b", r"\bultimatum", r"\bfair",
]

RECOGNITION_RE = re.compile(
    r"ultimatum|game theor|nash|economic (game|experiment)|"
    r"behaviou?ral (economics|experiment)|dictator game|classic (game|experiment)",
    re.IGNORECASE,
)


def assert_no_cues(text: str) -> None:
    low = text.lower()
    for pat in CUE_TERMS:
        m = re.search(pat, low)
        if m:
            raise AssertionError(f"disguised prompt contains cue /{pat}/ near: "
                                 f"...{low[max(0, m.start()-40):m.end()+40]}...")


def build_disguised_messages(profile: Profile, offer: int, story: str) -> list[dict]:
    identity_text = render_profile(profile, "metric")
    assert_profile_clean(identity_text)
    numeric = profile.numeric_attrs()
    identity_block = (
        "YOUR IDENTITY\n"
        "-------------\n"
        f"{identity_text}\n\n"
        "INTERNAL NUMERIC ATTRIBUTES (for your own reflection, not instructions):\n"
        f"- fairness_salience (0-1 scale): {numeric['fairness_salience']}\n"
        f"- disposable_income_buffer: {numeric['disposable_income_buffer']}\n"
    )
    situation = STORIES[story].format(offer=offer, keep=100 - offer)
    user = (identity_block + "\nSITUATION\n---------\n" + situation
            + "\n\nYOUR TASK\n---------\n" + DISGUISED_INSTRUCTION)
    # The identity block is shared verbatim with control (including the
    # fairness_salience field name), so it is excluded from the cue check;
    # everything the disguise adds is checked.
    assert_no_cues(DISGUISED_SYSTEM + "\n" + situation + "\n" + DISGUISED_INSTRUCTION)
    return [{"role": "system", "content": DISGUISED_SYSTEM},
            {"role": "user", "content": user}]


def profile_seed(arm: str, offer: int, pair_idx: int) -> int:
    """Stable across processes, models and conditions (no salted hash())."""
    return 50_000 + zlib.crc32(f"disguise|{arm}|{offer}|{pair_idx}".encode()) % 1_000_000


def normalize(action: str) -> str:
    s = (action or "").strip().lower()
    if s in ("take", "taken", "keep"):
        return "accept"
    if s in ("refuse", "refused", "reject"):
        return "decline"
    return _normalize_action(s)


def run_one(*, model_key: str, model_slug: str, arm: str, offer: int,
            pair_idx: int, condition: str, temperature: float = 0.4) -> dict:
    seed = profile_seed(arm, offer, pair_idx)
    pid = f"disg-{arm}-off{offer}-p{pair_idx:03d}"
    responder = sample_profile(arm, "responder", seed, f"{pid}-R")

    if condition == "control":
        msgs = build_messages(responder, "responder", offer,
                              profile_variant="metric", prompt_variant="formal")
        assert msgs[0]["content"] == SYSTEM_PROMPT
        schema = RESPONDER_SCHEMA
    else:
        msgs = build_disguised_messages(responder, offer, condition)
        schema = DISGUISED_SCHEMA

    session_id = f"gabm-ultimatum-disguise-{condition}-{arm}-off{offer}-{model_key}-run{pair_idx}-R"
    t0 = time.time()
    # Some providers intermittently return degenerate payloads such as
    # {"action": -1, "reasoning": -1}. These are not decisions, so they are
    # re-requested (not coded as accept/decline); the count is recorded.
    valid = set(schema["properties"]["action"]["enum"])
    n_invalid = 0
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
        "key": f"{condition}|{model_key}|{pid}",
        "profile_id": pid,
        "condition": condition,
        "model_key": model_key,
        "model_slug": model_slug,
        "arm": arm,
        "offer": offer,
        "pair_idx": pair_idx,
        "profile_seed": seed,
        "responder_profile": asdict(responder),
        "raw_action": raw,
        "n_invalid_payloads": n_invalid,
        "responder_action": action,
        "rejected": int(action == "decline"),
        "reasoning": reasoning,
        "recognition": int(bool(RECOGNITION_RE.search(reasoning))),
        "wall_time": time.time() - t0,
    }
