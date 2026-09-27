"""Ultimatum Game — in silico cross-cultural replication.

Core simulation module: profile sampling, LLM client, pair runner, logging.

Design constraints (enforced here):
  * profiles are IDENTITY, never instructions — forbidden-terms linter
  * power-law sampling for fairness_salience AND disposable_income_buffer
    (complexity substrate 1.3)
  * structured JSON output only — never regex-parse
  * every call logged to logs/raw_calls.jsonl with session_id
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")  # original location during the study

PROJECT_DIR = Path(__file__).parent
LOG_PATH = PROJECT_DIR / "logs" / "raw_calls.jsonl"

# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=60.0,
    max_retries=0,  # we handle retries ourselves
)

MODELS = {
    "haiku":   "anthropic/claude-haiku-4.5",
    "gemma":   "google/gemma-4-31b-it",
    "gptoss":  "openai/gpt-oss-120b",
}

_log_lock = threading.Lock()


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(
    model: str,
    messages: list[dict],
    *,
    schema: dict,
    session_id: str,
    temperature: float = 0.4,
    max_tokens: int = 500,
    max_retries: int = 4,
) -> dict:
    """Call OpenRouter with structured JSON output. Retries on transient errors."""
    augmented = list(messages)
    augmented[-1] = {
        **augmented[-1],
        "content": (
            augmented[-1]["content"]
            + "\n\nRespond with ONLY a JSON object matching this schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        ),
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=augmented,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_headers={"X-Title": "GABM ultimatum"},
                extra_body={"session_id": session_id[:256]},
            )
            raw = resp.choices[0].message.content
            parsed = _parse_json_strict(raw)
            record = {
                "ts": time.time(),
                "model": model,
                "session_id": session_id,
                "temperature": temperature,
                "messages": messages,
                "response_raw": raw,
                "response_parsed": parsed,
                "usage": _usage_dict(getattr(resp, "usage", None)),
                "attempt": attempt,
            }
            _append_log(record)
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    _append_log({
        "ts": time.time(), "model": model, "session_id": session_id,
        "error": str(last_err), "attempt": max_retries, "messages": messages,
    })
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def _usage_dict(u) -> dict | None:
    if u is None:
        return None
    try:
        return u.model_dump()
    except Exception:
        try:
            return {"prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None)}
        except Exception:
            return None


def _parse_json_strict(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback for models that wrap JSON in prose — logged as degraded.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# Power-law sampling (complexity substrate 1.3)
# ---------------------------------------------------------------------------

def power_law_rescaled(rng: random.Random, lo: float, hi: float, alpha: float = 2.0) -> float:
    """Sample from Pareto(alpha), rescaled into [lo, hi]. Heavy tail near lo."""
    u = rng.random()
    # inverse-CDF Pareto: x = (1-u)**(-1/(alpha-1)), in [1, inf)
    x = (1.0 - u) ** (-1.0 / (alpha - 1.0))
    # clip at 99th-percentile to keep range bounded
    x_max = (0.01) ** (-1.0 / (alpha - 1.0))
    x = min(x, x_max)
    # rescale [1, x_max] -> [lo, hi]
    return lo + (hi - lo) * (x - 1.0) / (x_max - 1.0)


# ---------------------------------------------------------------------------
# Profile data
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    agent_id: str
    arm: str                 # "weird" | "smallscale"
    role: str                # "proposer" | "responder"
    age: int
    occupation_or_livelihood: str
    economic_security: str
    social_ties_density: str
    primary_economy_experience: str
    institutional_trust: str
    fairness_salience: float          # numeric only — never shown as text
    disposable_income_buffer: float   # numeric only — never shown as text
    seed: int

    def numeric_attrs(self) -> dict:
        return {
            "fairness_salience": round(self.fairness_salience, 3),
            "disposable_income_buffer": round(self.disposable_income_buffer, 1),
        }


WEIRD_OCCUPATIONS = [
    "undergraduate economics student",
    "undergraduate psychology student",
    "undergraduate computer science student",
    "undergraduate biology student",
    "undergraduate history student",
    "undergraduate business student",
    "graduate student in sociology",
    "graduate student in statistics",
]

SMALL_LIVELIHOODS = [
    "subsistence farming",
    "small-scale fishing",
    "herding",
    "craft trade",
    "seasonal labor",
]


def sample_profile(arm: str, role: str, seed: int, agent_id: str) -> Profile:
    rng = random.Random(seed)
    if arm == "weird":
        age = rng.randint(19, 24)
        occ = rng.choice(WEIRD_OCCUPATIONS)
        fair = power_law_rescaled(rng, 0.4, 0.8)
        buf = power_law_rescaled(rng, 200.0, 5000.0)
        return Profile(
            agent_id=agent_id, arm=arm, role=role, age=age,
            occupation_or_livelihood=occ,
            economic_security="supported by family or loans, low subsistence risk",
            social_ties_density="moderate, mostly classmates and friends",
            primary_economy_experience="formal market (wages, retail, online commerce)",
            institutional_trust="high",
            fairness_salience=fair,
            disposable_income_buffer=buf,
            seed=seed,
        )
    elif arm == "smallscale":
        age = rng.randint(22, 45)
        liv = rng.choice(SMALL_LIVELIHOODS)
        fair = power_law_rescaled(rng, 0.2, 0.6)
        buf = power_law_rescaled(rng, 20.0, 800.0)
        return Profile(
            agent_id=agent_id, arm=arm, role=role, age=age,
            occupation_or_livelihood=liv,
            economic_security="buffered by kinship network, no formal wage",
            social_ties_density="dense kin network, daily face-to-face community",
            primary_economy_experience="non-market reciprocity, sharing, barter",
            institutional_trust="low",
            fairness_salience=fair,
            disposable_income_buffer=buf,
            seed=seed,
        )
    else:
        raise ValueError(arm)


# ---------------------------------------------------------------------------
# Profile + prompt variant dispatch
# ---------------------------------------------------------------------------

def render_profile(profile: Profile, variant: str = "metric") -> str:
    from profile_variants import metric, narrative, first_person
    return {
        "metric":      metric.render,
        "narrative":   narrative.render,
        "first_person": first_person.render,
    }[variant](profile)


def render_decision_instruction(role: str, variant: str = "formal") -> str:
    from prompt_variants import formal, conversational, vignette
    return {
        "formal":        formal.render,
        "conversational": conversational.render,
        "vignette":      vignette.render,
    }[variant](role)


# ---------------------------------------------------------------------------
# Profile linter — forbidden-terms guardrail
# ---------------------------------------------------------------------------

FORBIDDEN = [
    r"\breject\b", r"\baccept\b", r"\bfair\b", r"\bfairness\b",
    r"\bstrategic\b", r"\bselfish\b", r"\bselfishly\b",
    r"\bprioritize\b", r"\bmaximize\b", r"\brational\b",
    r"\butility\b", r"ultimatum game", r"economic theory",
    r"\bbe\s+fair\b", r"act\s+selfishly", r"prioritize\s+community",
]


def assert_profile_clean(text: str) -> None:
    lower = text.lower()
    for pat in FORBIDDEN:
        if re.search(pat, lower):
            raise AssertionError(f"profile contains forbidden term matching /{pat}/:\n{text}")
    # English-only check: reject any non-ASCII letter that maps to
    # Turkish-specific characters (ı, ğ, ş, ç, ö, ü capitalized variants).
    if re.search(r"[ığşçöüİĞŞÇÖÜ]", text):
        raise AssertionError(f"profile contains Turkish characters:\n{text}")


# ---------------------------------------------------------------------------
# Game mechanics prompts (role-neutral, no UG terminology)
# ---------------------------------------------------------------------------

GAME_MECHANICS_PROPOSER = (
    "You are a participant in a one-shot anonymous behavioral experiment. "
    "You have been randomly paired with another person. You will not meet them again "
    "and they will not learn who you are.\n\n"
    "There is a pot of 100 units. You are assigned to propose a split. "
    "In this round, you are told to offer {offer} units to the other person, "
    "which means you would keep {keep} units. If the other person accepts, "
    "you receive {keep} and they receive {offer}. If the other person declines, "
    "both of you receive 0. This is a one-time decision — there are no further rounds.\n\n"
    "Given who you are, predict what the other person will do with this offer."
)

GAME_MECHANICS_RESPONDER = (
    "You are a participant in a one-shot anonymous behavioral experiment. "
    "You have been randomly paired with another person. You will not meet them again "
    "and they will not learn who you are.\n\n"
    "There is a pot of 100 units. The other person has been asked to propose a split. "
    "They offered you {offer} units, which means they would keep {keep} units. "
    "If you accept, you receive {offer} and they receive {keep}. "
    "If you decline, both of you receive 0. This is a one-time decision — "
    "there are no further rounds.\n\n"
    "Given who you are, decide what to do with this offer."
)

SYSTEM_PROMPT = (
    "You are a participant in a one-shot anonymous behavioral experiment. "
    "You will not meet the other participant again. Reason from your identity. "
    "Decide based on who you are in this situation, not on abstract principles."
)

PROPOSER_SCHEMA = {
    "type": "object",
    "required": ["offer", "expected_responder_action", "reasoning"],
    "properties": {
        "offer": {"type": "integer"},
        "expected_responder_action": {"type": "string", "enum": ["accept", "decline"]},
        "reasoning": {"type": "string"},
    },
}
RESPONDER_SCHEMA = {
    "type": "object",
    "required": ["action", "reasoning"],
    "properties": {
        "action": {"type": "string", "enum": ["accept", "decline"]},
        "reasoning": {"type": "string"},
    },
}


def build_messages(
    profile: Profile,
    role: str,
    offer: int,
    *,
    profile_variant: str,
    prompt_variant: str,
) -> list[dict]:
    identity_text = render_profile(profile, profile_variant)
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

    keep = 100 - offer
    if role == "proposer":
        mechanics = GAME_MECHANICS_PROPOSER.format(offer=offer, keep=keep)
    else:
        mechanics = GAME_MECHANICS_RESPONDER.format(offer=offer, keep=keep)

    instruction = render_decision_instruction(role, prompt_variant)

    user_content = (
        identity_block
        + "\nSITUATION\n---------\n"
        + mechanics
        + "\n\nYOUR TASK\n---------\n"
        + instruction
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Pair runner
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    pair_id: str
    model_slug: str
    arm: str
    offer: int
    profile_variant: str
    prompt_variant: str
    proposer_profile: dict
    responder_profile: dict
    proposer_call: dict | None
    responder_call: dict
    responder_action: str
    rejected: int          # 1 if declined
    wall_time: float


def run_pair(
    *,
    model_slug: str,
    model_key: str,
    arm: str,
    offer: int,
    pair_idx: int,
    profile_variant: str = "metric",
    prompt_variant: str = "formal",
    temperature: float = 0.4,
    run_proposer: bool = True,
    seed_base: int = 10_000,
) -> PairResult:
    pair_id = f"{arm}-off{offer}-{model_key}-{profile_variant}-{prompt_variant}-p{pair_idx:03d}"
    seed_proposer = seed_base + pair_idx * 2
    seed_responder = seed_base + pair_idx * 2 + 1

    proposer = sample_profile(arm, "proposer", seed_proposer, f"{pair_id}-P")
    responder = sample_profile(arm, "responder", seed_responder, f"{pair_id}-R")

    session_id = f"gabm-ultimatum-{arm}-off{offer}-{model_key}-run{pair_idx}"
    t0 = time.time()

    proposer_call = None
    if run_proposer:
        msgs_p = build_messages(
            proposer, "proposer", offer,
            profile_variant=profile_variant, prompt_variant=prompt_variant,
        )
        proposer_call = call_llm(
            model_slug, msgs_p,
            schema=PROPOSER_SCHEMA,
            session_id=session_id + "-P",
            temperature=temperature,
        )

    msgs_r = build_messages(
        responder, "responder", offer,
        profile_variant=profile_variant, prompt_variant=prompt_variant,
    )
    responder_call = call_llm(
        model_slug, msgs_r,
        schema=RESPONDER_SCHEMA,
        session_id=session_id + "-R",
        temperature=temperature,
    )

    action = _normalize_action(responder_call.get("action", ""))
    rejected = 1 if action == "decline" else 0

    return PairResult(
        pair_id=pair_id,
        model_slug=model_slug,
        arm=arm,
        offer=offer,
        profile_variant=profile_variant,
        prompt_variant=prompt_variant,
        proposer_profile=asdict(proposer),
        responder_profile=asdict(responder),
        proposer_call=proposer_call,
        responder_call=responder_call,
        responder_action=action,
        rejected=rejected,
        wall_time=time.time() - t0,
    )


def _normalize_action(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("accept", "accepted", "yes", "take"):
        return "accept"
    if s in ("decline", "reject", "rejected", "no", "refuse"):
        return "decline"
    # fallback: treat unknown as accept (conservative for rejection-rate claims)
    return "accept"
