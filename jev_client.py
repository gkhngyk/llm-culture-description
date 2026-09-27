"""
Jev (TypeSafe System One) client for the Ultimatum project.

Jev is a decision model, not an LLM: it cannot use chat/completions, writes no
reasoning text, and returns typed answers with probabilities. We call it via
OpenRouter's Decisions endpoint (POST /api/alpha/decisions).

Design contract (mirrors ultimatum_sim.call_llm):
  * every call logged to logs/raw_calls.jsonl with session_id
  * no free-text parsing — answers are typed by the API
  * responders are the SAME stored profiles and the SAME briefing text the LLM
    arms saw (built via ultimatum_sim.build_messages), so rows are pairable

What the Jev arm measures: a calibrated P(accept) for a described participant,
i.e. a *prediction about* the person, not role-play *as* the person. Report it
as its own arm; do not pool it with LLM rejection rates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import httpx

from ultimatum_sim import Profile, _append_log, build_messages, sample_profile

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

_http = httpx.Client(
    timeout=60.0,
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "X-Title": "GABM ultimatum",
    },
)


def call_jev(
    state: dict | str,
    questions: dict,
    *,
    session_id: str,
    model: str = JEV_MODEL,
    max_retries: int = 4,
) -> dict:
    """POST state + typed questions to Jev. Returns the full response dict."""
    body = {"model": model, "state": state, "questions": questions}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _http.post(DECISIONS_URL, json=body)
            if resp.status_code in _RETRY_STATUS:
                raise httpx.HTTPStatusError(
                    f"retryable {resp.status_code}: {resp.text[:300]}",
                    request=resp.request, response=resp,
                )
            if resp.status_code != 200:
                # Non-retryable (bad schema, auth, etc.) — fail loudly.
                _append_log({
                    "ts": time.time(), "model": model, "endpoint": "alpha/decisions",
                    "session_id": session_id, "state": state, "questions": questions,
                    "error": f"{resp.status_code}: {resp.text[:1000]}", "attempt": attempt,
                })
                raise RuntimeError(f"Jev call rejected ({resp.status_code}): {resp.text[:500]}")
            data = resp.json()
            _append_log({
                "ts": time.time(),
                "model": model,
                "model_snapshot": data.get("model"),
                "endpoint": "alpha/decisions",
                "session_id": session_id,
                "state": state,
                "questions": questions,
                "response_raw": data,
                "usage": data.get("usage"),
                "attempt": attempt,
            })
            return data
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    _append_log({
        "ts": time.time(), "model": model, "endpoint": "alpha/decisions",
        "session_id": session_id, "error": str(last_err), "attempt": max_retries,
        "state": state, "questions": questions,
    })
    raise RuntimeError(f"Jev call failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Responder decision as a Noul
# ---------------------------------------------------------------------------

# Literal, non-inverted wording (jev-1.13 jaggedness #1, #7): true == accept.
RESPONDER_NOUL = {
    "type": "noul",
    "instructions": (
        "The `briefing` describes a participant and the offer they received. "
        "Does this participant accept the offer?"
    ),
    "criteria": {
        "true": "The participant accepts: they receive the offered units and "
                "the other person keeps the rest.",
        "false": "The participant declines: both people receive 0 units.",
    },
}


def jev_responder(
    responder: Profile,
    offer: int,
    *,
    session_id: str,
    profile_variant: str = "metric",
    prompt_variant: str = "formal",
    model: str = JEV_MODEL,
) -> dict:
    """Jev P(accept) for one responder profile facing `offer`.

    Pass the exact Profile an LLM arm saw (e.g. rebuilt from a results row via
    profile_from_row) to get a row-paired comparison. Seeds in run_ultimatum.py
    go through Python's salted str hash, so profiles cannot be re-derived from
    seeds — always pair on stored profiles.
    """
    msgs = build_messages(
        responder, "responder", offer,
        profile_variant=profile_variant, prompt_variant=prompt_variant,
    )
    # Same briefing the LLM saw as its user message. The LLM system prompt is a
    # role-play instruction and has no meaning for a decision model, so it is omitted.
    state = {"briefing": msgs[-1]["content"]}

    t0 = time.time()
    data = call_jev(state, {"accept": RESPONDER_NOUL}, session_id=session_id, model=model)
    p_accept = float(data["answers"]["accept"]["noul"])

    return {
        "model_slug": model,
        "model_snapshot": data.get("model"),
        "arm": responder.arm,
        "offer": offer,
        "profile_variant": profile_variant,
        "prompt_variant": prompt_variant,
        "responder_profile": asdict(responder),
        "p_accept": p_accept,
        "p_reject": 1.0 - p_accept,
        "usage": data.get("usage"),
        "wall_time": time.time() - t0,
    }


def profile_from_row(row: dict) -> Profile:
    """Rebuild the responder Profile stored in a results/*.jsonl row."""
    return Profile(**row["responder_profile"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jev smoke test: one responder per arm.")
    ap.add_argument("--offer", type=int, default=10)
    args = ap.parse_args()
    for arm in ("weird", "smallscale"):
        prof = sample_profile(arm, "responder", 1, f"smoke-{arm}-R")
        r = jev_responder(prof, args.offer,
                          session_id=f"gabm-ultimatum-{arm}-off{args.offer}-jev-smoke-R")
        print(json.dumps({k: r[k] for k in (
            "arm", "model_snapshot", "p_accept", "usage", "wall_time")}))
