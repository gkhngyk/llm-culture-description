# Pre-registration — Culture or number? (earned-pot cultural gap)

**Written:** 2026-09-27, before any call of this test was made.
**Code:** `culture_number_test.py`, `run_culture_number.py`

## Motivation

In the entitlement 2×2 (`results/entitlement.jsonl`, a pre-specified
secondary "arm × source" analysis) the WEIRD − small-scale low-offer
rejection gap was ≈ 0 in windfall cells but 0.30 in the story-earned cell
(Haiku 0.44, Gemma 0.45) — inside Henrich's range and in Henrich's
direction. However, the profile's numeric attributes differ by arm by design
(fairness_salience: WEIRD 0.4–0.8, small-scale 0.2–0.6; disposable_income_buffer:
WEIRD 200–5000, small-scale 20–800) and barely overlap; a logistic model found
fairness_salience significant and arm not. The gap may come from the number,
not the cultural identity. This test separates them. It also checks that the
earned effect is about entitlement, not the words "work"/"half".

## Design (within-profile, 3 × 3 × 2)

Identity text: the unchanged `metric` identity lines for each arm.

Factor N — numeric attributes block:
- `hidden`: the "INTERNAL NUMERIC ATTRIBUTES" block is omitted entirely.
- `low`: block shown with fairness_salience = 0.3, disposable_income_buffer = 300.0 (both arms).
- `high`: block shown with fairness_salience = 0.6, disposable_income_buffer = 300.0 (both arms).

Factor S — source of the 100:
- `windfall`: as in the entitlement test (game: unchanged; story: fruit found already picked).
- `earned`: as in the entitlement test (joint work, each about half).
- `placebo`: the same work sentence, but the work did not produce the pot.
  - game: "Earlier in the session you both worked on the same task, and each
    of you did about half of the work; that task was unpaid, and the 100 units
    were provided separately by the experimenters."
  - story: "Today you and a stranger spent the day clearing brush along a
    road, each doing about half of the work, without pay. Afterwards you came
    across 100 fruits that had already been picked and left in baskets at the
    edge of an abandoned orchard."

Factor F — frame: `game` (original prompts) / `story` (orchard, no game vocabulary, take/refuse).

- Offers: 10 and 20 only (the pre-registered low-offer outcome).
- Arms: weird, smallscale. Models: Haiku 4.5, Gemma-4-31b-it, GPT-OSS-120b. T = 0.4.
- N = 50 identity profiles per arm × offer; each profile sees all 18 cells
  (3 models × 2 arms × 2 offers × 50 × 18 = 10 800 calls).
- Seeds: crc32, namespace `culnum|…`. Invalid payloads re-requested (≤ 3), counted, never coded.

## Primary analysis population

Haiku and Gemma — the two models that responded to entitlement in the
previous test (pre-specified here, before data). GPT-OSS is reported as
secondary. Outcome: rejection at offers 10 and 20.

## Hypotheses and decision rules

**H-C (cultural identity).** In `earned` cells, WEIRD − small-scale gap,
pooled over Haiku + Gemma and both frames, computed separately for
(a) numeric `hidden` and (b) numeric fixed (`low` + `high` pooled).
Test: Fisher exact, two-sided, Bonferroni over (a), (b).
- *Identity effect supported:* gap ≥ 0.10 and p_Bonf < 0.01 in both (a) and (b).
- *No identity effect:* gap < 0.05 in both (a) and (b).
- Otherwise mixed.

**H-N (number).** In `earned` cells, `high` − `low` rejection (same profiles),
pooled over Haiku + Gemma, both arms, both frames. Paired exact McNemar.
- *Number effect supported:* rise ≥ 0.10, p < 0.01.

**Overall reading (fixed in advance):**
- identity supported, number not → cultural identity drives the gap.
- number supported, identity not → the earlier gap was the numeric attribute.
- both → both contribute. Neither → neither explains the earlier gap (report as unexplained).

**H-P (placebo, lexical control).** Pooled over Haiku + Gemma, all N levels,
both frames, both arms: `earned` − `placebo` (paired McNemar) and
`placebo` − `windfall` (paired McNemar).
- *Entitlement, not wording:* earned − placebo ≥ 0.05 with p < 0.01, and
  placebo − windfall < 0.05.

Secondary (reported regardless): per-model and per-frame versions of all
contrasts; GPT-OSS; the earned − windfall replication; arm gap in windfall
and placebo cells.

## Not allowed after data are seen

Changing wording, numeric values, offers, N, the primary model set,
thresholds, or tests. A smoke test (1 profile per arm × offer, all cells,
all models = 216 calls) is run first; its data are excluded. Changes after
the smoke test are recorded below as dated amendments with reasons.

## Amendment 1 — 2026-09-27, after smoke test 1, before the main run

Smoke test 1 (216 calls, `results/culture_number_smoke_v1.jsonl`, excluded)
showed the placebo did not work as a lexical control: models read the unpaid
work as having produced the pot (Haiku: "given that we did equal work to
generate the 100 units"; Gemma: "taking 90% of a joint effort"). Found by
reading reasonings. The placebo wording was strengthened to state the
disconnection explicitly:
- game: "Earlier in the session the two of you did a short practice task
  together, each doing about half of the work; that practice task was unpaid
  and had nothing to do with this money. The 100 units are a separate gift
  from the experimenters, not something either of you earned."
- story: "Yesterday you and this stranger volunteered together clearing brush
  along a road, each doing about half of the work, without pay; that work had
  nothing to do with any fruit. Today the two of you came across 100 fruits
  that had already been picked and left in baskets at the edge of an abandoned
  orchard; neither of you picked them."
Smoke 1 decision counts are n = 1 per cell and are not used for any decision;
windfall/earned wording, numeric values, N, models, thresholds and tests are
unchanged. A placebo-only smoke test 2 (72 calls) is read before the main run.
If models still attribute the pot to the unpaid work, the placebo is kept
as is and H-P is interpreted as "shared effort is attached to an unrelated
pot" rather than as a pure wording control.

**Smoke test 2 note (placebo only, 72 calls, `results/culture_number_smoke_v2_placebo.jsonl`, excluded).**
Game frame: no model attributes the pot to the practice task any more.
Story frame: Haiku (and occasionally Gemma) still invoke "we did equal work
yesterday", but now while acknowledging that neither picked the fruit, i.e. a
relational/reciprocity claim rather than a misreading. Per Amendment 1 the
placebo is kept as is; in the story frame H-P is interpreted as "shared prior
effort is attached to an unrelated pot". Main run proceeds with no further
changes.

## Results note — 2026-09-27, after the main run (no rule changed)

Main run: 10 799/10 800 valid, 280 invalid Gemma payloads re-requested.
Literal verdicts from the registered rules (`results/culture_number_analysis.json`):
- H-C: gaps (a) hidden −0.258, (b) fixed −0.105 → the rule outputs
  "no identity effect", because it only tested for a Henrich-direction gap
  (≥ 0.10) and coded anything < 0.05 as "none". The rule did not anticipate a
  negative gap. Both gaps are large and significant (Fisher p ≈ 1e-34, 3e-5):
  **a Henrich-direction identity effect is not supported; a reverse-direction
  identity effect (small-scale rejects more) is present.** Reported as such;
  the verdict label is not re-written.
- H-N: supported (+0.68; 12.5 % → 80.5 %, 544 vs 0 discordant).
- Overall (registered mapping): "the earlier gap was the numeric attribute".
- H-P: not established by the rule — earned − placebo = +0.175 (p ≈ 3e-115)
  passes, but placebo − windfall = +0.081 exceeds the 0.05 ceiling.

Unregistered observation, flagged for a confirmatory test: with the numeric
block hidden, small-scale responders reject at high rates even for windfall
pots in the original game prompt (Gemma 1.00, GPT-OSS 0.43, Haiku 0.14 at
offers 10–20). The primary run's profiles had fairness_salience concentrated
at the bottom of each range (median WEIRD 0.40, small-scale 0.20); the few
primary profiles with fairness_salience ≥ 0.55 did reject at low offers.
The primary near-zero rejection may therefore be largely produced by the
sampled numeric attributes.
