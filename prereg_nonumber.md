# Pre-registration — Study 2: primary replication without numeric attributes

**Written:** 2026-09-27, before any call of this study was made.
**Code:** `nonumber_test.py`, `run_nonumber.py`

## Why

The culture-or-number test (`prereg_culture_number.md`) showed that the
profile's numeric block — `fairness_salience` and `disposable_income_buffer`,
added in the original design for the classical-ABM sigmoid baseline and for
power-law heterogeneity — largely determines LLM decisions (earned pot:
fairness_salience 0.3 → 0.6 moved rejection 12.5 % → 80.5 %). Human
participants are not given a fairness score, and a field named "fairness"
contradicts the linter rule that bans the word from the identity text. The
original primary profiles had fairness_salience concentrated at the bottom of
each range (median WEIRD 0.40, small-scale 0.20). Study 2 repeats the primary
design with the numeric block removed. It is the paper's main test of the
Henrich replication; Study 1 (April primary, with numbers) is reported as the
methodological contrast.

## Design (identical to the primary run except as listed)

- Prompts: original system prompt, `metric` identity lines, `formal`
  decision instruction, original game mechanics — the only change is that the
  "INTERNAL NUMERIC ATTRIBUTES" block is removed from every prompt (proposer
  and responder).
- Models: `anthropic/claude-haiku-4.5`, `google/gemma-4-31b-it`,
  `openai/gpt-oss-120b`; T = 0.4.
- Arms: `weird`, `smallscale`. Offers: 10, 20, 30, 40, 50.
- N = 50 pairs per arm × offer × model; each pair = proposer call + responder
  call (3 000 calls).
- Seeds: crc32, namespace `nonum|…`, so the same identity profiles are used
  for all three models (fixes the salted-`hash()` issue of the primary run).
- Invalid payloads (e.g. Gemma `{"action": -1}`) re-requested up to 3 times,
  counted, never coded as a decision.

## Hypotheses (H1–H5 copied unchanged from the original pre-registration)

"<20 %" = offer 10, as in the original analysis.

| id | hypothesis | pass criterion |
|---|---|---|
| H1 | WEIRD rejection at offer 10 | ∈ [0.30, 0.70] |
| H2 | small-scale rejection at offer 10 | ∈ [0.00, 0.30] |
| H3 | WEIRD − small-scale gap at offer 10 | ≥ 0.15 AND χ² p < 0.01 Bonferroni (k = 5 offers) AND Cohen h ≥ 0.3 |
| H4 | dose-response within each arm | Spearman ρ(offer, rejection) < −0.9 (rejection falls as offer rises) |
| H5 | cross-model robustness | max |model_rate − 3-model mean| ≤ 0.15 in every arm × offer cell |

Finding decision rule (unchanged): H1 ∧ H2 ∧ H3 → Henrich replication.
All pooled over the three models unless stated.

(H4 note: the original table wrote "ρ > 0.9"; the intended and analysed
direction was decreasing rejection with offer, i.e. ρ < −0.9. Stated
explicitly here to avoid ambiguity; no other change.)

## Added before data (new)

- **H-R (reverse gap).** small-scale − WEIRD at offer 10 ≥ 0.15 AND χ²
  p < 0.01 (Bonferroni k = 5) AND Cohen h ≥ 0.3. Motivated by the
  culture-or-number test, where hidden-number small-scale responders rejected
  more. Reported as "reverse-direction cultural gap".
- **H-S1 (numbers suppressed rejection).** Pooled rejection at offer 10 in
  Study 2 exceeds Study 1 (`results/primary.jsonl`) by ≥ 0.10, Fisher exact
  p < 0.01. Between-study comparison, run five months apart; the disguise
  test's control arm (same prompts as Study 1, run 2026-09-26) is reported
  alongside as a drift check.
- Secondary, reported regardless: per-model H1–H3 and H-R; proposer
  predicted-rejection by arm; blind theme coding is **not** re-run in this
  study.

## Not allowed after data are seen

Changing prompts, N, offers, models, thresholds, or tests. A smoke test
(1 pair per arm × offer × model, 60 calls) is run first; its data are
excluded; any change after it is recorded below as a dated amendment.

**Smoke test note — 2026-09-27, before the main run.** 30 pairs (60 calls,
`results/nonumber_smoke.jsonl`, excluded): 60/60 valid, 0 invalid payloads,
reasonings read for confounds, none found. No changes; main run proceeds as
registered.

## Results note — 2026-09-27 (rules applied as registered)

1 500/1 500 pairs, 32 invalid payloads re-requested. Offer-10 rejection,
3 models pooled: WEIRD 0.060, small-scale 0.607.
H1 fail, H2 fail, H3 fail → no Henrich replication (finding rule false).
H-R supported pooled (gap −0.547, p_Bonf 5e-23, h = −1.29) and in each model
(Gemma −0.82, GPT-OSS −0.54, Haiku −0.28). H4: small-scale yes, WEIRD no
(floor). H5 fail (max deviation 0.51). H-S1 supported: offer-10 rejection
0.033 (Study 1) → 0.333 (Study 2), p = 2e-23; drift check (Study-1 prompts
re-run 2026-09-26) 0.013.
