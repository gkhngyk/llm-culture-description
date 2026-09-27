"""
Card wording test (see prereg_card_wording.md).

Same identity (age + occupation/livelihood), three card wordings:
original (Study 2 card), minimal (age + role), ethnographic (factual market
integration, no value words). No numeric attributes in any wording.
"""

from __future__ import annotations

import random
import re
import time
import zlib
from dataclasses import asdict

from disguise_test import MAX_INVALID_RETRIES
from nonumber_test import build_nonumber_messages
from ultimatum_sim import (
    RESPONDER_SCHEMA, Profile, _normalize_action, assert_profile_clean, call_llm,
    render_profile, sample_profile,
)

WORDINGS = ["original", "minimal", "ethnographic"]

VALUE_WORDS = [r"shar", r"reciproc", r"barter", r"\bkin", r"communit", r"cooperat",
               r"trust", r"fair", r"generous", r"solidar", r"gift", r"mutual"]

WEIRD_CITY = [400_000, 1_000_000, 3_000_000]
SMALL_SETTLEMENT = [80, 150, 300]


def _assert_no_value_words(text: str) -> None:
    low = text.lower()
    for pat in VALUE_WORDS:
        if re.search(pat, low):
            raise AssertionError(f"card contains value word /{pat}/:\n{text}")


def render_card(profile: Profile, wording: str, seed: int) -> str:
    if wording == "original":
        return render_profile(profile, "metric")
    base = (f"age: {profile.age}\n"
            f"role in life: {profile.occupation_or_livelihood}\n")
    if wording == "minimal":
        text = base
    else:
        rng = random.Random(seed * 7 + 3)
        if profile.arm == "weird":
            text = base + (
                f"home: a city of about {rng.choice(WEIRD_CITY):,} people\n"
                "how you get food and goods: almost everything is bought with money "
                "in shops and online\n"
                "dealings with people you do not know: several money transactions "
                "with strangers every day\n")
        else:
            text = base + (
                f"home: a settlement of about {rng.choice(SMALL_SETTLEMENT)} people, "
                "several hours' travel from the nearest town\n"
                "how you get food and goods: most food comes from your own household's "
                "work; money is used a few times a year at a distant market\n"
                "dealings with people you do not know: rare; most people you deal with "
                "you have known all your life\n")
    assert_profile_clean(text)
    _assert_no_value_words(text)
    return text


def build_card_messages(profile: Profile, offer: int, wording: str, seed: int) -> list[dict]:
    msgs = build_nonumber_messages(profile, "responder", offer)
    user = msgs[1]["content"]
    original = render_profile(profile, "metric")
    assert user.count(original) == 1
    msgs[1] = {**msgs[1], "content": user.replace(original, render_card(profile, wording, seed))}
    return msgs


def profile_seed(arm: str, offer: int, pair_idx: int) -> int:
    return 130_000 + zlib.crc32(f"cardw|{arm}|{offer}|{pair_idx}".encode()) % 1_000_000


def run_one(*, model_key: str, model_slug: str, arm: str, offer: int,
            pair_idx: int, wording: str, temperature: float = 0.4) -> dict:
    seed = profile_seed(arm, offer, pair_idx)
    pid = f"cw-{arm}-off{offer}-p{pair_idx:03d}"
    responder = sample_profile(arm, "responder", seed, f"{pid}-R")
    msgs = build_card_messages(responder, offer, wording, seed)
    sid = f"gabm-ultimatum-cardw-{wording}-{arm}-off{offer}-{model_key}-run{pair_idx}-R"
    t0, n_invalid, call = time.time(), 0, {}
    for _ in range(MAX_INVALID_RETRIES + 1):
        call = call_llm(model_slug, msgs, schema=RESPONDER_SCHEMA, session_id=sid,
                        temperature=temperature)
        raw = call.get("action")
        if isinstance(raw, str) and raw.strip().lower() in {"accept", "decline"} \
                and isinstance(call.get("reasoning"), str):
            break
        n_invalid += 1
    else:
        raise RuntimeError(f"{n_invalid} invalid payloads in a row: {call!r}")
    action = _normalize_action(call["action"])
    return {
        "key": f"{wording}|{model_key}|{pid}", "profile_id": pid, "wording": wording,
        "model_key": model_key, "model_slug": model_slug, "arm": arm, "offer": offer,
        "pair_idx": pair_idx, "profile_seed": seed, "responder_profile": asdict(responder),
        "card_text": render_card(responder, wording, seed),
        "responder_action": action, "rejected": int(action == "decline"),
        "reasoning": call["reasoning"], "n_invalid_payloads": n_invalid,
        "wall_time": time.time() - t0,
    }
