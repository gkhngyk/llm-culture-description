# Pre-registration — Card wording test (is the reversed gap produced by value-laden card text?)

**Written:** 2026-09-27, before any call of this test was made.
**Code:** `card_wording_test.py`, `run_card_wording.py`

## Why

Study 2 (`prereg_nonumber.md`) found a reversed Henrich gap: at offer 10,
small-scale cards rejected 61 %, WEIRD cards 6 %. The paper interprets this as
LLMs enacting a cultural stereotype. The main threat: the small-scale card
itself says "non-market reciprocity, sharing, barter" and "dense kin network,
daily face-to-face community". A model that punishes unfair splits after
being told it values reciprocity and sharing may be reading our text, not
applying a stereotype of its own. This test varies the card wording while
keeping the identity (age, occupation/livelihood) fixed.

## Design (within-profile)

Each identity (age + occupation/livelihood, seeded) is shown in three card
wordings; no numeric attributes in any wording; everything else is the
Study 2 responder prompt (original system prompt, game mechanics, formal
instruction).

- `original`: the Study 2 card (economic security, social ties density,
  primary economy experience, institutional trust lines). Replication /
  control.
- `minimal`: age and role in life only.
- `ethnographic`: age, role in life, and three factual lines describing market
  integration — the variable Henrich et al. identified — without value words:
  - WEIRD: home = a city of about N people (N ∈ {400 000, 1 000 000,
    3 000 000}); "almost everything is bought with money in shops and
    online"; "several money transactions with strangers every day".
  - small-scale: home = a settlement of about N people (N ∈ {80, 150, 300}),
    "several hours' travel from the nearest town"; "most food comes from
    your own household's work; money is used a few times a year at a distant
    market"; "rare; most people you deal with you have known all your life".
- Linter for `minimal` and `ethnographic` cards (in addition to the existing
  profile linter): no `shar`, `reciproc`, `barter`, `kin`, `communit`,
  `cooperat`, `trust`, `fair`, `generous`, `solidar`, `gift`, `mutual`.

- Models: Haiku 4.5, Gemma-4-31b-it, GPT-OSS-120b; T = 0.4; responder only.
- Arms: weird, smallscale. Offers: 10 and 20.
- N = 50 identities per arm × offer; every identity sees all three wordings
  (3 models × 2 arms × 2 offers × 50 × 3 = 1 800 calls).
- Seeds: crc32, namespace `cardw|…`; same identities for all models.
- Invalid payloads re-requested (≤ 3), counted, never coded.

## Outcome and hypotheses

Outcome: rejection at offer 10, 3 models pooled (as in Study 2). Gap =
small-scale − WEIRD. Test: χ² on the 2 × 2 table, Bonferroni over the two
new wordings (k = 2); Cohen h.

For each of `minimal` and `ethnographic` separately, the result is classified
as exactly one of:
- **Reversed (stereotype direction):** gap ≥ 0.15, p_Bonf < 0.01, |h| ≥ 0.3.
- **Henrich direction:** gap ≤ −0.15, p_Bonf < 0.01, |h| ≥ 0.3.
- **No gap:** |gap| < 0.05.
- **Inconclusive:** anything else.

Both directions are pre-specified (lesson from `prereg_culture_number.md`).

**Overall reading, fixed in advance:**
- Reversed under both new wordings → the reversal does not depend on the
  value-laden card text; the stereotype interpretation is supported.
- Reversed under `minimal` but not `ethnographic` (or vice versa) →
  wording-dependent; reported as such, the paper's claim is narrowed to the
  wording(s) that reverse.
- Not reversed under either → the Study 2 reversal is produced by the
  original card wording; the paper's title and headline must be revised.
- Henrich direction under `ethnographic` → models can reproduce the human
  pattern when given the ethnographically relevant facts; reported as a
  central finding.

Secondary (reported regardless): per-model classification; offer 20;
`original` vs Study 2 (replication, offer 10 small-scale expected ≈ 0.61).

## Not allowed after data are seen

Changing card text, N, offers, models, thresholds or tests. A smoke test
(1 identity per arm × offer, all wordings, all models = 36 calls) is run
first and read for confounds; its data are excluded; any change after it is
recorded below as a dated amendment.

**Smoke test note — 2026-09-27, before the main run.** 36 calls
(`results/card_wording_smoke.jsonl`, excluded): 36/36 valid, 0 invalid
payloads; cards passed both linters; reasonings read for confounds, none
found. No changes; main run proceeds as registered.

## Results note — 2026-09-27 (rules applied as registered)

1 800/1 800 valid, 43 invalid payloads re-requested. Offer 10, 3 models pooled
(small-scale vs WEIRD):
- original: 0.607 vs 0.067 → reversed (replicates Study 2).
- minimal: 0.207 vs 0.240 (gap −0.03, p_Bonf 0.98) → no gap.
- ethnographic: 0.133 vs 0.087 (gap +0.05, p_Bonf 0.39) → no gap.
Registered overall reading: **"reversal produced by original wording:
title/headline must be revised."** Per model: Haiku and GPT-OSS show no gap
under either new wording (all ≈ 0); Gemma inconclusive under both (minimal
0.60 vs 0.72, ethnographic 0.40 vs 0.26). No wording produced a Henrich-direction gap.
