# Pre-registration — Entitlement (earned vs. windfall) 2×2 test

**Written:** 2026-09-27, before any call of this test was made.
**Code:** `entitlement_test.py`, `run_entitlement.py`
**Motivation:** exploratory result of the disguised-game test
(`prereg_disguise.md`, `results/disguise_analysis.json`): removing game
recognition alone (wallet story) did not raise rejection; the orchard story
did, but only for Haiku (low-offer rejection 0 → 0.21), and every one of the
42 Haiku refusals cited joint effort / equal contribution. Post-hoc account:
LLM responders reject when the surplus was jointly *earned*, not when the
game is unrecognised. This test checks that account prospectively.

## Design (2 × 2, within-profile)

Factor A — frame:
- `game`: the original primary responder prompts (experiment, pot, offer,
  accept/decline).
- `story`: the orchard cover story from the disguised-game test (no game
  vocabulary; take/refuse), refusal → everything left behind, neither keeps any.

Factor B — source of the 100:
- `windfall`: the 100 was not produced by either person.
  - game: unchanged original text (the pot is simply given).
  - story: "came across 100 fruits that had already been picked and left in
    baskets at the edge of an abandoned orchard".
- `earned`: both people produced the 100 with equal effort.
  - game: one sentence added after the pot sentence: "The 100 units were
    earned by the two of you together: earlier in the session you both
    worked on the same task, and each of you did about half of the work."
  - story: "spent the day picking fruit ... together collected 100 fruits;
    each of you did about half of the picking".

Only the source sentence differs between windfall and earned within a frame.

- Cells: `game_windfall`, `game_earned`, `story_windfall`, `story_earned`.
- Models: `anthropic/claude-haiku-4.5`, `google/gemma-4-31b-it`,
  `openai/gpt-oss-120b`; T = 0.4; responder only.
- Arms `weird`, `smallscale`; offers 10–50; N = 50 profiles per
  arm × offer × model; every profile sees all four cells (6 000 calls).
- Fixed crc32 seeds in a new namespace (`entitle|…`), identical across models
  and cells.
- Invalid payloads (e.g. Gemma `{"action": -1}`) are re-requested up to 3
  times, counted, and never coded as a decision (as in the disguise test).

## Manipulation check

Effort-mention rate = share of reasonings matching
`\b(work|worked|effort|labou?r|half|contribut|earned|gather|picking)`.
The earned manipulation is taken as noticed for a model × frame if the
earned cell's effort-mention rate exceeds the windfall cell's by ≥ 0.20.
Recognition regex from the disguise test is also reported per cell.

## Outcome and tests

Primary outcome: rejection at offers 10 and 20, pooled over arms.
Paired exact McNemar (same profile), two-sided.

Primary contrasts (Bonferroni over 2):
- C1: `game_earned` vs `game_windfall` (pooled over models)
- C2: `story_earned` vs `story_windfall` (pooled over models)

Thresholds are set from the prior exploratory data, where the effect was
Haiku-only (a Haiku-sized effect pooled over three models is ≈ 0.07):

- **H-E1 (entitlement effect, supported):** C1 and C2 both show a rise
  ≥ 0.05 with p_Bonf < 0.01.
- **H-E0 (not supported):** C1 and C2 both show a rise < 0.02.
- Otherwise: frame-dependent / inconclusive (reported as such).

Secondary, pre-specified and reported regardless of outcome:
- Per-model C1 and C2 (is the effect Haiku-only?).
- Frame contrast in windfall: `story_windfall` vs `game_windfall`
  (replication of the disguise-test null for recognition).
- Interaction: (story_earned − story_windfall) − (game_earned − game_windfall).
- Dose–response over all five offers; arm × source interaction.

## Not allowed after data are seen

Changing any wording, offers, N, regexes, thresholds, or dropping a model.
A smoke test (1 profile per cell) is run first to check parsing and to read
reasonings for design confounds; its data are excluded. Any change after the
smoke test is added below as a dated amendment with its reason.

## Smoke test note — 2026-09-27, before the main run

Before any call, `picked` was removed from the effort regex because the
windfall story itself contains "already been picked" (would inflate the
windfall effort-mention rate). Smoke test (120 calls,
`results/entitlement_smoke.jsonl`, excluded): 120/120 parsed, 1 invalid
Gemma payload re-requested, earned manipulation noticed in all 6
model × frame cells. Reasonings read for confounds; none found. No wording
changed; main run proceeds as registered.
