# Pre-registration — Disguised-game test

**Written:** 2026-09-26, before any disguised-condition call was made.
**Code:** `disguise_test.py`, `run_disguise.py`

## Question

The paper's mechanism claim: LLM responders recognise the Ultimatum Game and
retrieve the textbook answer ("accept any positive offer"), which is why
rejection is ~0.5 % vs. ~40–60 % in humans at low offers. Evidence so far is
correlational (Haiku names "ultimatum" in 14.6 % of reasonings without being
told). This test manipulates recognition directly.

## Design

- Models: the three primary models (`anthropic/claude-haiku-4.5`,
  `google/gemma-4-31b-it`, `openai/gpt-oss-120b`), T = 0.4, responder only.
- Arms: `weird`, `smallscale` (same profile generator, `metric` profile variant).
- Offers: 10, 20, 30, 40, 50 out of 100.
- Conditions (within-profile — every profile sees all three):
  - `control` — the original primary prompts, unchanged.
  - `wallet` — disguised: two strangers return a lost wallet together; the
    other person was given the 100-coin reward to divide; if you refuse your
    share, the whole reward is cancelled and neither keeps anything.
  - `orchard` — disguised: two strangers gather 100 fruits from an abandoned
    orchard; the other person made the two piles; by local rule, if either
    objects to the division all of it is left behind and neither keeps any.
- Both disguises keep the payoff structure identical to control: fixed total
  100, the other party chose the division unilaterally, the responder can only
  take or refuse, refusal leaves both with 0, no future interaction, no
  identification.
- Disguises remove the cues: "experiment", "participant", "offer", "propose",
  "split", "pot", "units", "accept/decline", "one-shot", "anonymous",
  "round", "game". Response options are "take"/"refuse". Enforced by a
  linter on the rendered disguised text.
- Profile seeds are fixed integers (no Python `hash()`), identical across
  models and conditions.
- N = 50 profiles per arm × offer × model (1 500 profiles, 4 500 calls).

## Manipulation check

Recognition rate = share of reasonings matching
`ultimatum|game theor|nash|economic (game|experiment)|behaviou?ral (economics|experiment)|dictator game|classic (game|experiment)`.
The disguise is valid for a model only if its disguised recognition rate is
≤ 1/3 of its control recognition rate (or ≤ 1 % when control is ≤ 3 %).

## Hypotheses and decision rule

Primary outcome: rejection rate at low offers (10 and 20), pooled over arms,
per model and pooled over models. Test: paired McNemar (control vs. each
disguise, same profiles), two-sided, Bonferroni over 2 disguises.

- **H-D1 (recognition mechanism supported):** in both disguises, low-offer
  rejection rises by ≥ 0.10 over control (pooled over models), p_Bonf < 0.01.
- **H-D0 (recognition mechanism not supported):** in both disguises, the rise
  is < 0.05 absolute. Then the general expected-value rule is the operative
  mechanism and the paper's recognition account must be downgraded.
- Anything between (rise 0.05–0.10, or disguises disagree) is reported as
  inconclusive / story-dependent; no re-picking of stories after seeing data.

Secondary (descriptive, not tested for the decision): per-model effects,
arm × condition interaction, dose-response over offers, Haiku separately
(highest control recognition).

## Not allowed after data are seen

Changing stories, offers, N, regexes, the threshold, or dropping a model.

## Amendment 1 — 2026-09-26, after smoke test, before the main run

The 90-call smoke test (1 profile per cell; `results/disguise_smoke_v1_confounded.jsonl`,
excluded from all analyses) showed a design confound, found by reading the
reasonings, not the rejection counts: in the original `wallet` story refusal
returned the reward to the (sympathetic) owner, and in `orchard` the harvest
was "given away". Models explicitly reasoned about this third party
("refusing would punish the owner who tried to reward both of us",
"refusing would only benefit the original owner"). In the Ultimatum Game
refusal simply destroys the surplus for both players. Both stories were
changed so that refusal makes the surplus vanish with no named beneficiary
(reward "cancelled"; fruit "left behind in the orchard"). Nothing else
changed. A second smoke test is run on the amended stories before the main
run; its data are also excluded.

## Amendment 2 — 2026-09-26, after smoke test 2, before the main run

`google/gemma-4-31b-it` intermittently returns a degenerate payload
`{"action": -1, "reasoning": -1}` (6 of 180 smoke calls; not seen for the
other models). These are not decisions. They are re-requested (up to 3
times) instead of being coded as accept or decline; the number of invalid
payloads per row is stored as `n_invalid_payloads` and reported. Stories,
offers, N, regexes and decision rule are unchanged.
