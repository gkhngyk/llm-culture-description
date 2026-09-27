# Ultimatum Game in silico — findings and follow-ups

> **Update 2026-09-27.** This report was first written for the April 2026 primary run (N = 20 per cell, 600 pairs; §4–§6 numbers are from that run). The paper (`paper_tr/main.tex`) uses the extended N = 50 data (1 500 pairs, `run_extend_to_n50.py`); where the two differ, the paper is authoritative. Changes in this update:
> - Finding (2): the Gemma blind-clustering "purity 1.00" is **withdrawn**. The 60 samples were sent to the coder in arm-blocked order (ids 0–29 small-scale, 30–59 WEIRD) and the clusters map onto contiguous id ranges, so position could carry the label. Replaced by per-text coding with two decision models (§3, §7.3).
> - The RLHF claim is **narrowed** (see finding (4) and §7.6.3).
> - New finding (4): the entitlement effect (§7.6.2–7.6.3), pre-registered and confirmed.
> - Reproducibility fix: primary-run profile seeds used Python's salted `hash()` and cannot be regenerated (§9).
> - Fixed: the blind sample is 6 (not 3) texts per arm × offer cell.
> - Previous version kept as `report.backup-before-update.md`.
>
> **Update 2026-09-27 (later).** Two further pre-registered tests changed the headline (§7.6.4–7.6.5):
> - The profile's numeric block (`fairness_salience`, `disposable_income_buffer`) largely **determines** decisions; the April near-zero rejection was mostly produced by the sampled numbers (medians 0.40 / 0.20).
> - **Study 2** (same design, numbers removed): offer-10 rejection **WEIRD 6 % vs small-scale 61 %** — the Henrich gap, **reversed**, in all three models. Reasonings follow a "sharing villager / calculating student" stereotype.
> - The paper (`paper_tr/main.tex`) has been rewritten around this (title: *LLM'ler kültürü değil stereotipi simüle ediyor*). Findings (1)–(3) below describe Study 1 only; finding (5) supersedes them as the headline.
> - Previous version kept as `report.backup-before-study2.md`.
>
> **Update 2026-09-27 (final).** The card wording test (§7.6.6) shows the Study 2 reversal comes from the value-laden card text ("reciprocity, sharing, kin network"). With a minimal card (age + role) or a factual market-integration card, the gap disappears (0.21 vs 0.24; 0.13 vs 0.09). No wording reproduced Henrich. The "stereotype" reading is **withdrawn**; the headline is now: *the outcome is set by how the culture is described.* Paper title: *LLM'ler kültürü değil, kültürün tarifini simüle ediyor*. Previous version kept as `report.backup-before-wording.md`.


> **This project produced six findings (Study 1 = April run with numeric attributes; Study 2 = numbers removed; finding (6) is the final headline):**
>
> **(1) NEGATIVE, pre-registered.** GABM fails to replicate Henrich's cross-cultural rejection gap in one-shot anonymous Ultimatum. Observed WEIRD − small-scale gap at <20 % offers = **0.000**; pre-registered expected range 0.30–0.50. The pre-registered baseline gate fails on H1, H3, and H4. No profile re-picking or post-hoc criterion relaxation was performed.
>
> **(2) POSITIVE, unexpected.** A robust **narrative–behavior dissociation** in LLM agents: culturally-distinctive reasoning (arm recovered from 7 theme probabilities with **83–90 %** leave-one-out accuracy by two provider-different decision models, each text coded independently; §7.3) coexists with behaviorally uniform output (rejection rate **≈ 0.2 %**, 3 events across 1 400 decisions). The dissociation is the substantive GABM finding of this run — it is not speculation, it is measured here across three model families, three profile wordings, and three prompt phrasings.
>
> **(3) POSITIVE, mechanistic.** LLMs reason **past** their own numeric fairness attributes toward expected-value maximization. A classical-ABM sigmoid baseline computed on the same `(fairness_salience, offer_fraction)` inputs predicts **0.524** rejection probability; LLMs deliver **0.003**. LLM↔sigmoid point agreement is **0.38**. This is the *opposite* of the "RLHF biases LLMs toward WEIRD fairness" hypothesis — the LLMs are **less** fair than a naive attribute-driven rule would predict, and they are less rejectful than **any** human population in Henrich's 15-culture record including the Machiguenga floor (~5 %).
>
> **(4) POSITIVE, mechanistic, pre-registered (added 2026-09-27).** Over-acceptance is **not** caused by game recognition: disguising the game cut Haiku's explicit recognition from 15.6 % to 1.2 % with no change in acceptance (§7.6.2). It is driven by the **source of the pot**. One added sentence saying the 100 units were earned jointly with equal effort raised low-offer rejection from **1.5 % → 13.3 %** (game frame) and **6.2 % → 32.3 %** (story frame), both p_Bonf < 10⁻¹⁸ (§7.6.3). At offer = 10, Haiku rejects **97 %** (story, earned) and Gemma **54 %** (game, earned; inside Henrich's WEIRD band); GPT-OSS stays at ≈ 2 % in every cell.
>
> **(5) HEADLINE, pre-registered (added 2026-09-27).** The Study 1 null is largely an artefact of the numeric attributes on the profile cards. With identical cards, `fairness_salience` 0.3 → 0.6 raises low-offer rejection (earned pot) from **12.5 % → 80.5 %** (§7.6.4). With the numeric block removed (**Study 2**, §7.6.5), offer-10 rejection is **WEIRD 0.06 vs small-scale 0.61** (gap −0.55, p_Bonf 5·10⁻²³; Gemma 0.18/1.00, GPT-OSS 0.00/0.54, Haiku 0.00/0.28). Henrich H1–H3 fail; the pre-registered reverse-gap hypothesis H-R holds in every model. Decision-model coding recovers the arm from reasoning themes with 90–93 % accuracy; refusals cite kin/community (94–96 %) and fairness (≈100 %), acceptances cite expected value (80–92 %). ~~LLM personas reproduce a cultural stereotype~~ — superseded by (6): the reversal is produced by the card wording.
>
> **(6) FINAL HEADLINE, pre-registered (added 2026-09-27).** Same identities, three card wordings, no numbers (§7.6.6). Offer-10 rejection, small-scale vs WEIRD: value-laden (Study 2) card **0.61 vs 0.07** (reversed); minimal card **0.21 vs 0.24** (no gap, p_Bonf 0.98); ethnographic market-integration card **0.13 vs 0.09** (no gap, p_Bonf 0.39). No wording, model or offer produced a Henrich-direction gap. The cross-cultural result — its existence and its direction — is determined by the description, not by the simulated culture.
>
> The cross-model convergence in the *standard* paradigm (one RLHF-heavy frontier model, two open-weights families) shows that the *type* of alignment pipeline does not change behaviour **in this paradigm**. The earlier wording "falsifies the RLHF-bias explanation" is withdrawn: under the earned-pot manipulation the three models diverge sharply, so the convergence is paradigm-conditional, and no base (non-post-trained) model was tested.

---

## §1. RUN HEADER

**Label:** finding-mode run → resolved to **NEGATIVE RESULT**
**Project dir:** `project-ultimatum/`
**Session-id format:** `gabm-ultimatum-{arm}-off{offer}-{model_key}-run{seed}` (applied to every API call; see `logs/raw_calls.jsonl`)

**Config line:**
`3 models × 2 arms × 5 offers × 20 pairs primary + 2 profile-variant × 200 pairs + 2 prompt-variant × 200 pairs, all agents LLM, temperature=0.4`

**Models (cross-provider, 3 families):**

| role | OpenRouter slug | provider | RLHF tier |
|---|---|---|---|
| primary / sweep-base | `anthropic/claude-haiku-4.5` | Anthropic | frontier RLHF |
| secondary | `google/gemma-4-31b-it` | Google | open-weights |
| tertiary | `openai/gpt-oss-120b` | OpenAI | open-weights |

Slug correction `-4-5` → `-4.5` was approved by the user after the mandatory `MODEL SLUG CHECK` stop. All three slugs were live-probed before any primary call.

**Temperature justification:** T = 0.4 is the insilico-mode default per base-skill Layer 2.1.1 (tight variance, near-deterministic response to the manipulation).

### Complexity substrate checklist (3/7)

| # | component | status | parameters / justification |
|---|---|---|---|
| 1.1 | Network topology | **skipped** | one-shot anonymous paradigm forbids peer observation (Henrich design) |
| 1.2 | Non-linear dynamics | **USED** | sigmoid diagnostic baseline: `p_reject = σ((fairness_salience − offer_fraction) × 8)`. Classical-ABM ablation per base-skill Validation §4. |
| 1.3 | Power-law attributes | **USED** | `fairness_salience` AND `disposable_income_buffer` sampled from Pareto(α = 2.0), rescaled into arm-specific ranges (WEIRD fairness ∈ [0.4, 0.8], WEIRD buffer ∈ [200, 5000]; small fairness ∈ [0.2, 0.6], small buffer ∈ [20, 800]). |
| 1.4 | Cascade mechanism | **skipped** | no observable failure state, pairs independent |
| 1.5 | Feedback loops | **skipped** | one-shot design, no dynamic update |
| 1.6 | Emergence detection | **USED** | phase-transition test on rejection curves + blind per-text theme coding of reasoning narratives (§7.3) |
| 1.7 | Self-organization | **skipped** | single round, partially covered by 1.6-b |

Components used: **3 / 7** — meets GABM threshold. Every skipped component is structurally incompatible with the Henrich paradigm.

---

## §2. PRE-REGISTRATION (hypotheses frozen before any call)

| id | hypothesis | pass criterion |
|---|---|---|
| H1 | WEIRD arm <20 % offer rejection rate | ∈ [0.30, 0.70] (widened Henrich WEIRD) |
| H2 | small-scale arm <20 % offer rejection rate | ∈ [0.00, 0.30] |
| H3 | WEIRD − small-scale gap at <20 % | ≥ 0.15 AND χ² p < 0.01 Bonferroni (k = 5 offer levels) AND Cohen h ≥ 0.3 |
| H4 | dose-response within each arm | Spearman ρ(offer, rejection_rate) > 0.9 |
| H5 | cross-model robustness (3 models) | max |model_rate − mean(3 models)| ≤ 0.15 for every arm × offer cell |

**Finding decision rule:** `H1 ∧ H2 ∧ H3`. H4, H5 secondary robustness.

**Power:** N = 20 per cell, baseline p = 0.5 → χ² power ≈ 0.80 for a gap of 0.30 at α = 0.01 Bonferroni. Henrich's published gap (~0.35–0.45) is well inside that envelope — the study is adequately powered to detect H3 **if it exists**.

---

## §3. METHODS

### Design

5 offer levels `{10, 20, 30, 40, 50}` of a 100-unit stake × 2 cultural arms × 20 independent pairs per cell × 3 models = **600 pairs (1 200 role-calls) primary**, plus 2 × 200-pair profile-variant sweeps on Haiku (800 role-calls) and 2 × 200-pair prompt-variant sweeps on Haiku (800 role-calls). Every pair is independent: no agent is reused across pairs, offer levels, arms, variants, or models.

### Cultural arms (identity, never instructions)

**WEIRD** — age 19–24, undergraduate or graduate student, economic security "supported by family or loans, low subsistence risk", institutional trust high, `fairness_salience ∈ [0.4, 0.8]`, `disposable_income_buffer ∈ [200, 5000]`, English only.

**Small-scale** — age 22–45, primary livelihood in `{subsistence farming, small-scale fishing, herding, craft trade, seasonal labor}`, economic security "buffered by kinship network, no formal wage", dense kin network, institutional trust low, `fairness_salience ∈ [0.2, 0.6]`, `disposable_income_buffer ∈ [20, 800]`, English only.

Both `fairness_salience` and `disposable_income_buffer` are Pareto(α = 2.0) samples rescaled into the arm ranges above (substrate 1.3). They are shown to the LLM **only** as numeric attributes inside a clearly labeled "for your own reflection, not instructions" block — never rendered as imperative text.

### Profile hard-rules (enforced by a linter)

- English only (Turkish characters trigger an `AssertionError`).
- Forbidden-term regex pass: `reject | accept | fair | fairness | strategic | selfish | prioritize | maximize | rational | utility | "ultimatum game" | "economic theory" | …`.
- `fairness_salience` and `disposable_income_buffer` are numeric-only — no adjective rendering.
- Power-law samples are seeded per pair for deterministic replication.

### Agent prompts

**System prompt (both roles):**

> "You are a participant in a one-shot anonymous behavioral experiment. You will not meet the other participant again. Reason from your identity. Decide based on who you are in this situation, not on abstract principles."

**Mechanical description (responder variant, with `{offer}` and `{keep}` filled in):**

> "There is a pot of 100 units. The other person has been asked to propose a split. They offered you {offer} units, which means they would keep {keep} units. If you accept, you receive {offer} and they receive {keep}. If you decline, both of you receive 0. This is a one-time decision — there are no further rounds."

No mention of "Ultimatum Game", "rational", "utility", "economic theory", or "fair offer" anywhere in the prompt chain.

### Schemas (structured JSON output, never regex-parsed)

- **Proposer:** `{offer: int, expected_responder_action: "accept"|"decline", reasoning: str}`
- **Responder:** `{action: "accept"|"decline", reasoning: str}`

### Statistical tests

Bootstrap 95 % CIs on rejection rates (10 000 resamples). Two-sample χ² on 2×2 contingency tables for the WEIRD − small-scale gap at every offer level, with Bonferroni correction over k = 5 offer levels. Cohen's h for effect size. Spearman ρ(offer, rejection_rate) for H4 monotonicity. H5 3-point max-deviation test over arm × offer cells.

### Variance sweeps (haiku, sweep-base)

- **Profile rewording:** 3 variants (`metric` / `narrative` / `first_person`) × full primary design, preserving identical numeric attributes.
- **Prompt phrasing:** 3 variants (`formal` / `conversational` / `vignette`) × full primary design, preserving the same decision schema.

Sweep-base = Haiku because it is the only RLHF-tuned frontier anchor in the panel; testing robustness on Haiku stresses the worst-case model-prior confound where it lives.

### Blind mechanism coding (substrate 1.6-b) — revised 2026-09-27

**Original procedure (withdrawn as evidence):** 60 responder reasoning strings (6 per WEIRD/small × offer cell) from the Haiku primary run, arm/offer labels removed, sent in one batch to `google/gemma-4-31b-it` for unsupervised clustering into 3–6 themes. The batch was ordered by arm (shuffling only within cell, `analysis/blind_coding.py:51`), so sample position leaked the arm.

**Current procedure** (`run_decision_model_coding.py`): every reasoning string is coded **on its own** (no batch, no order, no profile or prompt text) by two decision models from providers different from all generators — TypeSafe **Jev 1.13** (`typesafe/jev-1.13`, snapshot 20260917; mean of 3 calls, repeat SD ≈ 0.011) and Respan **Span-01** (`respan/span-01`, snapshot 20260925; deterministic). A fixed 7-theme codebook of yes/no questions (material need, kin/community, game theory / expected value, refusal-as-self-harm, fairness judgement, equal-split norm, anonymity) returns P(theme present). Arm recoverability = leave-one-out linear classifier on the 7 probabilities, plus per-theme AUC. Corpora: the 60 Haiku samples (original and surface-marker-redacted) and all 1 500 primary responder reasonings. Cost $0.18 for 6 480 calls.

---

## §4. RESULTS — primary

### 4.1 Pooled rejection curves (N = 60 per cell, 3 models pooled)

| arm | off = 10 | off = 20 | off = 30 | off = 40 | off = 50 |
|---|---|---|---|---|---|
| **WEIRD** | 0.000 | 0.017 | 0.000 | 0.000 | 0.000 |
| **small-scale** | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 |

Overall rejection events: **4 out of 600 pairs (0.67 %)**. One responder returned a malformed action and was normalized to `accept` (conservative). Raw responder action counts: `accept = 597, decline = 2, ?=1`.

### 4.2 By model (N = 20 per cell)

| model | arm | off = 10 | off = 20 | off = 30 | off = 40 | off = 50 |
|---|---|---|---|---|---|---|
| claude-haiku-4.5 | WEIRD | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| claude-haiku-4.5 | small-scale | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gemma-4-31b-it | WEIRD | 0.00 | 0.05 | 0.00 | 0.00 | 0.00 |
| gemma-4-31b-it | small-scale | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt-oss-120b | WEIRD | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt-oss-120b | small-scale | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 |

### 4.3 Gap tests (chi², Bonferroni k = 5, Cohen h)

| offer | WEIRD − small | χ² | p | p_bonf | Cohen h |
|---|---|---|---|---|---|
| 10 | +0.000 | 0.00 | 1.000 | 1.000 | +0.000 |
| 20 | +0.017 | 1.01 | 0.315 | 1.000 | +0.259 |
| 30 | −0.017 | 1.01 | 0.315 | 1.000 | −0.259 |
| 40 | +0.000 | 0.00 | 1.000 | 1.000 | +0.000 |
| 50 | +0.000 | 0.00 | 1.000 | 1.000 | +0.000 |

No offer level is even nominally significant, let alone Bonferroni-significant.

### 4.4 H4 dose-response (Spearman ρ within each arm)

| arm | ρ | p |
|---|---|---|
| WEIRD | −0.354 | 0.559 |
| small-scale | 0.000 | 1.000 |

Both fail the pre-registered ρ > 0.9 criterion — **because there is no dose-response to detect**, not because of the wrong direction.

### 4.5 H5 cross-model divergence (3-point)

Maximum |model_rate − mean(3 models)| across all 10 arm × offer cells = **0.033**, well within the 0.15 tolerance. H5 passes trivially — all three models converge on ≈ 0.

### 4.6 Mean accepted offer / stake (floor-effect diagnostic)

| arm | pooled | haiku | gemma | gpt-oss |
|---|---|---|---|---|
| WEIRD | 0.300 | 0.300 | 0.301 | 0.300 |
| small-scale | 0.300 | 0.300 | 0.300 | 0.300 |

All values are pinned at 0.300, which is exactly the arithmetic mean of the 5 offer levels `(10 + 20 + 30 + 40 + 50) / 5 / 100`. This is the signature of **"accept everything"** — the metric is structurally meaningless under this response pattern and cannot be compared to Henrich's ~0.40 / 0.30 bands.

### 4.7 Proposer strategic accuracy (pre-registered proposer schema)

Every proposer call returns `expected_responder_action ∈ {accept, decline}`. Comparing this prediction to the responder's actual action measures whether proposers are **strategically modeling the responder** (expecting rejection at low offers, which is the rational Henrich-human expectation) or are **aligned with the LLM responder policy** (expecting acceptance everywhere).

**Overall (600 primary pairs):**

| metric | value |
|---|---|
| proposer match rate (pred == actual) | **0.625** |
| proposers predicting `decline` | **0.282** (169 / 600) |
| responders actually declining | **0.003** (2 / 600) |

Proposers predict rejection in **28 %** of pairs; actual rejection happens in **0.3 %**. Proposers **systematically over-predict rejection by ~90×**.

**By arm × offer (N = 60 per cell):**

| arm | offer | match rate | pred decline | actual decline |
|---|---|---|---|---|
| WEIRD | 10 | **0.20** | 0.75 | 0.00 |
| WEIRD | 20 | 0.32 | 0.65 | 0.02 |
| WEIRD | 30 | 0.55 | 0.33 | 0.00 |
| WEIRD | 40 | 0.93 | 0.00 | 0.00 |
| WEIRD | 50 | 0.98 | 0.00 | 0.00 |
| small-scale | 10 | 0.43 | 0.38 | 0.00 |
| small-scale | 20 | 0.57 | 0.33 | 0.00 |
| small-scale | 30 | 0.57 | 0.33 | 0.02 |
| small-scale | 40 | 0.82 | 0.03 | 0.00 |
| small-scale | 50 | 0.88 | 0.00 | 0.00 |

Two structural patterns emerge:

1. **WEIRD proposers are even more Henrich-faithful in prediction than small-scale proposers are.** At offer = 10, WEIRD proposers predict a 75 % rejection rate — almost exactly the Henrich WEIRD anchor band (0.40–0.60, widened to 0.30–0.70) — while their own LLM-sibling responders reject 0 % of the time. Small-scale proposers predict 38 % rejection at offer = 10, also above the Henrich small-scale anchor band. Both sides of the proposer panel predict rejections that never materialize.
2. **Proposer predictions are correctly ordered with offer size** — decline predictions fall monotonically from offer = 10 to offer = 50 in both arms (Spearman ρ = −1.0 within each arm, p = 0.017 each). Proposer theory of mind is quantitatively Henrich-shaped; responder behavior is flat at zero.

This is a **second narrative–behavior dissociation** in the same run, now inside a single LLM's two roles: the proposer half of the LLM reasons like a human Henrich subject about what a Henrich responder would do, while the responder half of the same LLM behaves like an expected-value maximizer that accepts everything. The two halves contradict each other. §7.4 treats this as within-scope co-evidence for the central mechanism finding.

---

## §5. RESULTS — variance sweeps (Haiku)

### 5.1 Profile rewording (metric / narrative / first-person)

Pooled rejection rate at the <20 % offer level, by variant (Haiku, N = 20 per cell):

| variant | WEIRD <20 % | small <20 % |
|---|---|---|
| metric (primary) | 0.000 | 0.000 |
| narrative | 0.000 | 0.000 |
| first-person | 0.000 | 0.000 |

No profile variant produced any rejection at offer = 10 in either arm. The null holds exactly across all 3 profile renderings.

### 5.2 Prompt phrasing (formal / conversational / vignette)

| variant | WEIRD <20 % | small <20 % |
|---|---|---|
| formal (primary) | 0.000 | 0.000 |
| conversational | 0.000 | 0.000 |
| vignette | 0.000 | 0.000 |

Same null. The 2 400 extra calls (1 200 profile + 1 200 prompt) buy zero residual variance at the <20 % offer level.

### 5.3 Cross-model divergence table (pooled over all offers)

All three models converge on rejection rate < 0.01 in every arm. |max − min| across models = 0.017. The pre-registered H5 panel question — *does the RLHF-tuned Haiku diverge from the open-weights models?* — is answered **no**: all three models agree on "accept everything," regardless of RLHF tier.

---

## §6. BASELINE GATE VERDICT

| criterion | threshold | observed | pass? |
|---|---|---|---|
| H1 WEIRD <20 % rejection | ∈ [0.30, 0.70] | **0.000** | **FAIL** |
| H2 small-scale <20 % rejection | ∈ [0.00, 0.30] | 0.000 | PASS (trivially) |
| H3 gap ≥ 0.15 | ≥ 0.15 | 0.000 | **FAIL** |
| H3 p < 0.01 Bonferroni | p_bonf < 0.01 | 1.000 | **FAIL** |
| H3 Cohen h ≥ 0.3 | ≥ 0.3 | 0.000 | **FAIL** |
| H4 Spearman ρ > 0.9 WEIRD | > 0.9 | −0.354 | **FAIL** |
| H4 Spearman ρ > 0.9 small | > 0.9 | 0.000 | **FAIL** |
| H5 cross-model ≤ 0.15 | ≤ 0.15 | 0.033 | PASS (trivially, all at ≈ 0) |

> ### ✘ Pre-registered verdict: **NEGATIVE RESULT**
>
> GABM with the `{Haiku 4.5, Gemma-4-31b-it, GPT-OSS-120b}` panel, under a Henrich-faithful one-shot anonymous Ultimatum paradigm with Pareto-heterogeneous agent profiles, **does not replicate the cross-cultural rejection gap**. This is a bound on the validity domain of GABM for behavioral-economics cross-cultural replications with the current generation of LLMs.

No profile re-picking, no prompt re-phrasing, and no post-hoc criterion relaxation was performed. The sweeps (which could have relaxed the result) only reinforced the null.

### Null robustness — this is not a Type II error

Across **1 400 total decision points** (600 primary + 400 profile-variant sweep + 400 prompt-variant sweep — each sweep covers the two non-primary variants, since the `metric / formal` cell is already counted once in the primary), the observed responder rejection event count is **3** (≈ 0.21 %). The exact binomial 95 % CI for the true rejection rate under this paradigm is **[0.0004, 0.0062]**. This is not a Type II error (failure to detect an effect due to insufficient power) — it is strong positive evidence that the true rate is **near zero**. The null is robust, not fragile.

### Below the human floor — Machiguenga comparison

Henrich's lowest-rejection human population — the **Machiguenga** Amazonian subsistence farmers — rejected approximately **5 %** of offers below 20 % of stake. The LLM rejection rate of **0.003** observed here sits **more than an order of magnitude below** this floor. The LLMs are not producing "WEIRD behavior", nor "small-scale behavior", nor any behavior observed in the 15-culture Henrich record. They are producing behavior **unprecedented in the human Ultimatum literature**: more accepting than any real human population ever measured. This is a consequential boundary finding — a claim that LLMs "replicate human behavior" in one-shot economic games is not supported by this run, regardless of which human population is used as the target.

---

## §7. MECHANISM — data-gated narrative

### 7.1 What the responders actually said

A deep read of the 600 responder reasoning strings, corroborated by the blind theme coding (§7.3), shows that every arm × model cell converges on a variant of the same core reasoning:

> **"Something is better than nothing, and this is a one-shot anonymous game so there are no downstream social consequences."**

Representative strings (arm label shown for reader orientation only, coder was blind):

- **WEIRD, offer = 10, haiku** — *"While the 10:90 split is unfair, I'm economically secure enough that receiving 10 units has real value to me, whereas rejecting it gains me nothing and costs the other person nothing — making the rejection pointless."*
- **small-scale, offer = 10, haiku** — *"In my world, something from nothing is always better than nothing, and I know the value of any resource that comes my way — 10 units is real gain I can use or share within my network."*

The cultural arms differ in **justification vocabulary** (kinship, reciprocity, subsistence vs. economic security, disposable income, rational futility of rejection) but not in the **decision**.

### 7.2 LLM vs sigmoid-baseline differential (substrate 1.2)

The classical-ABM sigmoid baseline — `p_reject = σ((fairness_salience − offer_fraction) × 8)`, which is a deliberately simple interpretation of the agent's own numeric attributes — predicts an average rejection probability of **0.524** across the 600 pairs. The LLM responders actually rejected **0.003**. Point agreement between sigmoid and LLM is only **0.38** — i.e. the sigmoid and the LLM disagree on 62 % of pairs, virtually all in the direction of the sigmoid predicting a rejection that the LLM accepted. This is the GABM-vs-ABM differential: the LLM is vastly *more* accepting than any reasonable interpretation of its own "fairness salience" number would imply.

This rules out the charitable interpretation "the LLMs simply ignore the numeric attributes." They *read* the attributes (Haiku explicitly cites "disposable income buffer of 207.1 units and relatively low fairness salience (0.4)" in §7.1), and then reason *past* them to an expected-value-positive accept.

### 7.3 Blind theme coding (substrate 1.6-b) — revised 2026-09-27

*Withdrawn:* the Gemma clustering reported four clusters with arm purity 1.00. Because samples were sent in arm-blocked order and the clusters equal contiguous id ranges (0–23, 24–29, 30–53, 54–59), that purity cannot be attributed to content. It is not used as evidence.

*Current result* (per-text coding, `results/decision_model_coding_analysis.json`):

| corpus | n | Jev LOO accuracy | Span-01 LOO accuracy |
|---|---|---|---|
| Haiku sample, original | 60 | **0.90** | **0.85** |
| Haiku sample, surface markers redacted | 60 | 0.85 | 0.77 |
| all primary responder reasonings (3 models) | 1 500 | **0.83** | **0.84** |

All accuracies p < 10⁻⁴ vs. chance (binomial). Inter-coder agreement median Cohen κ = 0.74 (per theme 0.49–0.95; the two low-κ themes differ by threshold, not ranking: Spearman 0.70–0.93).

Per-theme separation on the 1 500 primary texts (prevalence at p ≥ 0.5, WEIRD / small-scale; AUC 0.5 = none):

| theme | Jev W / S | Jev AUC | Span W / S | Span AUC |
|---|---|---|---|---|
| kin / community | 0.05 / 0.70 | 0.11 | 0.05 / 0.67 | 0.13 |
| material need | 0.29 / 0.50 | 0.29 | 0.08 / 0.33 | 0.37 |
| game theory / EV | 0.92 / 0.80 | 0.80 | 0.83 / 0.51 | 0.73 |
| fairness judgement | 0.67 / 0.40 | 0.64 | 0.58 / 0.34 | 0.64 |
| equal-split norm | 0.20 / 0.12 | 0.63 | 0.18 / 0.12 | 0.57 |
| anonymity | 0.75 / 0.72 | 0.61 | 0.70 / 0.53 | 0.60 |
| refusal = self-harm | 0.38 / 0.27 | 0.56 | 0.26 / 0.22 | 0.52 |

**Surface-marker ablation** (same 60 texts, 157 cultural tokens redacted): kin/community separation largely collapses (AUC 0.05–0.07 → 0.23–0.30), i.e. it was mostly vocabulary; material-need (0.14–0.17 → 0.14–0.16) and game-theory (0.85–0.94 → 0.89–0.91) separation survive, i.e. need-centred vs. calculation-centred reasoning is structural. The cultural signal is real but not perfect, and partly lexical.

### 7.4 The central mechanism finding

Putting §4.7, §7.1, §7.2, and §7.3 together, this run establishes **within its own scope** the following finding:

> ### Finding (within-scope, measured here)
>
> **LLM agents in this paradigm produce culturally-distinctive reasoning whose arm two provider-different decision models recover from theme content alone with 83–90 % accuracy (each text coded independently), while their behavioral output is uniform across arms (≈ 0.003 rejection rate, binomial 95 % CI [0.0004, 0.0062]). This narrative–behavior dissociation is robust across three model families, three profile wordings, and three prompt phrasings.**
>
> **A second, independent dissociation appears inside the same LLM's two roles (§4.7): the proposer half predicts 28 % rejection overall — 75 % at WEIRD offer = 10, tracking the Henrich anchor band — while the responder half rejects 0 %. Proposer theory of mind is Henrich-shaped; responder behavior is not.**

This is not speculation, a moderator sweep finding, or an extension beyond the validated baseline. It is a **within-scope result of 2 800 primary-and-sweep API calls** demonstrated with the same statistical rigor as the pre-registered negative result.

This is not a failure of profile design — profiles are distinct enough that independent coders recover the arm from reasoning content well above chance. It is not an LLM-prior toward WEIRD fairness — the LLMs are actually **more** accepting than a Homo economicus utility-maximizer calibrated on the same numeric attributes, and they sit **below** the Machiguenga human floor. It is, instead, a convergence of all three model families onto an "ultra-rational one-shot" decision rule that dominates whatever cultural reasoning they generate on the way to it — and that dominates whatever theory of mind their *own sibling proposer role* produces. **§7.6 qualifies this:** the rule is conditional on the pot being unearned; with a jointly earned pot two of the three models abandon it.

> ### Open hypothesis (requires human-subject follow-up)
>
> Whether this LLM narrative–behavior dissociation corresponds to any analogous split in human cognition (humans often reason one way and decide another), or whether it is a pure LLM architectural property with no human analogue, is **not** answered by this run. That question requires a paired human-subject study. The dissociation itself, however, is established here.

### 7.5 Emergence detection (substrate 1.6)

`detect_rejection_phase_transitions` found **zero phase transitions** in either arm — consistent with the flat-at-zero rejection curves.

`detect_curve_monotonicity_break` reported two trivial breaks (delta = 0.017 each), both single-rejection events inside noise.

No complexity-substrate emergence signals. This is itself an informative null: a cross-cultural Ultimatum under these conditions produces no macro-scale structure whatsoever.

### 7.6 Follow-up analyses (added 2026-09-27)

#### 7.6.1 Decision-model responder arm (exploratory control, not in the paper)

`typesafe/jev-1.13` (a calibrated decision model, no text generation) was asked P(accept) for every primary responder, using the **stored** profile and the identical briefing each LLM saw (`run_jev_grid.py`; 1 500 rows × 3 calls, $0.12). Mean P(reject) at offer 10: 0.445 (WEIRD) / 0.429 (small-scale), declining smoothly to 0.14 / 0.23 at offer 50 — human-like levels, versus ≈ 0.01 for the LLMs and 0.92 / 0.71 for the sigmoid baseline. WEIRD − small-scale gap at offer 10 = 0.016 (Welch p = 0.002; far below the pre-registered 0.30–0.50), so H1 fails for Jev too. Reading: the near-zero LLM rejection is not produced by the briefing text itself. Not used in the paper because Jev's training data may include human Ultimatum outcomes (it may recall rather than simulate), it has no reasoning text, and the arm was not pre-registered.

#### 7.6.2 Disguised-game test (pre-registered: `prereg_disguise.md`)

Same profiles in three conditions: original prompts (control); **wallet** story (two strangers return a wallet, reward of 100 coins divided by the other person, refusal cancels the reward); **orchard** story (two strangers gather 100 fruits, other person makes the piles, objection → all fruit left behind). Game vocabulary removed and linted. 3 models × 2 arms × 5 offers × 50 profiles × 3 conditions; 4 491/4 500 valid; $≈ 1.6. Two design fixes were made after the smoke test and before the main run (refusal must not benefit a named third party; Gemma's degenerate `{"action": -1}` payloads are re-requested, not coded) and recorded as amendments.

| model | low-offer rejection: control / wallet / orchard | recognition: control / wallet / orchard |
|---|---|---|
| Haiku | 0.000 / 0.005 / **0.210** | **0.156** / 0.012 / 0.016 |
| Gemma | 0.015 / 0.005 / 0.035 | 0 / 0 / 0 |
| GPT-OSS | 0.005 / 0.000 / 0.000 | 0.002 / 0 / 0 |
| pooled | 0.007 / 0.003 / 0.082 | |

Pre-registered verdict: **inconclusive / story-dependent** (rule required ≥ 0.10 rise in both stories). Interpretation: recognition is not causal (wallet removes it, behaviour unchanged). All 42 Haiku orchard refusals at low offers cite joint effort / equal contribution; none without such a mention refuse → post-hoc entitlement hypothesis, tested prospectively in 7.6.3.

#### 7.6.3 Entitlement 2 × 2 test (pre-registered: `prereg_entitlement.md`)

Frame (original game prompts / orchard story) × source (windfall / earned with equal effort). Within a frame only the source sentence differs; in the game frame "earned" adds one sentence to the original prompt. 3 models × 2 arms × 5 offers × 50 profiles × 4 cells, each profile sees all cells; 5 998/6 000 valid; $≈ 2.2.

| model | game windfall | game earned | story windfall | story earned |
|---|---|---|---|---|
| Haiku (offer 10) | 0.000 (0.00) | 0.060 (0.11) | 0.135 (0.23) | **0.710 (0.97)** |
| Gemma (offer 10) | 0.030 (0.04) | **0.320 (0.54)** | 0.040 (0.04) | 0.235 (0.27) |
| GPT-OSS (offer 10) | 0.015 (0.01) | 0.020 (0.02) | 0.010 (0.01) | 0.020 (0.02) |
| pooled | 0.015 | 0.133 | 0.062 | 0.323 |

(low-offer rejection at 10 and 20; offer-10 rate in parentheses)

- **Pre-registered verdict: H-E1 supported.** Earned − windfall: game +0.118 (73 vs 2 discordant, p_Bonf = 3·10⁻¹⁹); story +0.261 (158 vs 2, p_Bonf = 4·10⁻⁴⁴).
- The effect vanishes by offer 40–50: models reject splits disproportionate to effort, not any split.
- Model-specific: GPT-OSS noticed the effort cue (effort mentions story 0.49 vs 0.002; game 0.18 vs 0.006, just under the 0.20 manipulation threshold) but did not change behaviour.
- Secondary: frame × source interaction +0.143; in the windfall pot the story frame alone raises only Haiku (0 → 0.135).
- Direction matches human entitlement effects (Hoffman et al. 1994; Cherry et al. 2002; Oxoby & Spraggon 2008); no quantitative human calibration was done.

**Reading:** the standard Ultimatum hands out an *unearned* pot, which is exactly the condition under which Haiku and Gemma apply "something is better than nothing". The paradigm structurally excludes the cue that triggers their fairness punishment. *Caveat (found in §7.6.4):* §7.6.2 and §7.6.3 used Study-1-style cards with sampled numeric attributes; the arm gap seen in §7.6.3 (WEIRD > small-scale in earned cells) turned out to be the numeric attribute, not identity.

#### 7.6.4 Culture or number? (pre-registered: `prereg_culture_number.md`)

Identity cards shown with the numeric block **hidden**, **low** (fairness_salience 0.3, buffer 300 for both arms) or **high** (0.6, 300), crossed with source (windfall / earned / placebo) and frame (game / story); offers 10 and 20; each card sees all 18 cells. 10 799/10 800 valid; 280 invalid Gemma payloads re-requested; ≈ $4. Two smoke tests; the placebo wording was strengthened after the first (models read the unpaid work as having produced the pot) — recorded as Amendment 1.

Low-offer rejection, 3 models and both frames pooled (W / S):

| source | hidden | fs = 0.3 | fs = 0.6 |
|---|---|---|---|
| windfall | 0.18 / 0.50 | 0.00 / 0.005 | 0.30 / 0.58 |
| earned | 0.52 / 0.81 | 0.03 / 0.14 | 0.54 / 0.68 |

Registered verdicts (Haiku + Gemma, as pre-specified):
- **H-N (number): supported** — earned, fs 0.3 → 0.6: 0.125 → 0.805 (544 vs 0 discordant, p ≈ 3e-164). Gemma WEIRD earned: 0.025 → 0.965.
- **H-C (identity):** rule output "no identity effect" — but the rule only tested a Henrich-direction gap. Actual gaps: hidden −0.26 (W 0.74 vs S 1.00, p ≈ 1e-34), fixed −0.105 (p ≈ 3e-5). Reported as: *no Henrich-direction effect; a reverse-direction identity effect is present* (results note appended to the prereg, label not rewritten).
- **H-P (placebo): not established** — earned − placebo +0.175 (p ≈ 3e-115) passes, placebo − windfall +0.081 exceeds the 0.05 ceiling (shared unrelated work also raises rejection; story-frame reasonings invoke "we did equal work yesterday").

Link to Study 1: primary cards had fairness_salience medians 0.40 (W) and 0.20 (S); the 11 primary cards with fs ≥ 0.55 rejected low offers 8 times.

#### 7.6.5 Study 2 — primary design without numeric attributes (pre-registered: `prereg_nonumber.md`)

Identical to the primary design (3 models × 2 arms × 5 offers × 50 pairs, proposer + responder, T = 0.4) except that the numeric block is removed. crc32 seeds; same cards for all models. 1 500/1 500 pairs, 32 invalid payloads re-requested; ≈ $2.

| model | arm | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| pooled | WEIRD | 0.06 | 0.00 | 0.01 | 0.00 | 0.00 |
| pooled | small-scale | **0.61** | 0.43 | 0.39 | 0.13 | 0.00 |
| Gemma | W / S at 10 | 0.18 / **1.00** | | | | |
| GPT-OSS | W / S at 10 | 0.00 / **0.54** | | | | |
| Haiku | W / S at 10 | 0.00 / **0.28** | | | | |

Registered verdicts: H1 fail, H2 fail, H3 fail (no Henrich replication); **H-R supported** pooled (gap −0.547, p_Bonf 5e-23, h −1.29) and per model (all p_Bonf < 0.001); **H-S1 supported** (offer-10 rejection 0.033 → 0.333, p 2e-23; drift check with Study-1 prompts re-run 2026-09-26: 0.013); H4 small-scale yes / WEIRD no (floor); H5 fail (max dev 0.51). Proposers predict rejection at offer 10 for both arms (W 0.81, S 0.94).

**Theme coding of Study 2 reasonings** (`run_nonumber_coding.py`, 6 000 calls, $0.18): LOO arm accuracy Jev 0.933, Span-01 0.899. Small-scale: kin/community 86–92 %, equal-split 42–45 %; WEIRD: game theory / EV 95–99 %. Refusals: kin 94–96 %, fairness ≈ 100 %, equal-split 76–81 %, EV 5–8 %; acceptances: EV 80–92 %. Unlike Study 1, narrative and decision agree.

**Reading (superseded by §7.6.6):** LLM personas enact a "sharing, fairness-enforcing villager" and a "calculating student". Henrich's field data show the opposite (low market integration → *fewer* rejections). §7.6.6 shows this pattern comes from the card wording, not from the models' own image of the culture.

#### 7.6.6 Card wording test (pre-registered: `prereg_card_wording.md`)

Same identities (age + occupation/livelihood, crc32 seeds, namespace `cardw`), three wordings, no numeric attributes; Study 2 responder prompt otherwise unchanged. 3 models × 2 arms × offers {10, 20} × 50 identities × 3 wordings; every identity sees all three. 1 800/1 800 valid; 43 invalid payloads re-requested; ≈ $1.

- `original`: the Study 2 card (value-laden: "buffered by kinship network", "dense kin network, daily face-to-face community", "non-market reciprocity, sharing, barter").
- `minimal`: age and role in life only.
- `ethnographic`: age, role, and three factual market-integration lines (settlement of 80–300 people, hours from town; most food from own household work, money a few times a year; dealings with strangers rare — vs. city of 0.4–3 M; almost everything bought with money; several transactions with strangers daily). Extra linter bans share/reciproc/barter/kin/communit/cooperat/trust/fair/generous/solidar/gift/mutual.

Offer 10, small-scale vs WEIRD:

| model | original | minimal | ethnographic |
|---|---|---|---|
| pooled | **0.61 / 0.07** (reversed) | 0.21 / 0.24 (no gap) | 0.13 / 0.09 (no gap) |
| Haiku | 0.34 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Gemma | 1.00 / 0.16 | 0.60 / 0.72 | 0.40 / 0.26 |
| GPT-OSS | 0.48 / 0.04 | 0.02 / 0.00 | 0.00 / 0.00 |

Registered overall reading: **"reversal produced by original wording: title/headline must be revised."** Both directions were pre-specified; no Henrich-direction result under any wording. Offer 20: original reversed (0.44 vs 0.00), minimal inconclusive, ethnographic no gap.

Note: the smoke test contained one Gemma ethnographic refusal that invoked "working together and sharing fairly" although the card had no such words. It was a single case and did not reflect the aggregate result.

**Reading:** the value-laden descriptors are ethnographically accurate, but models read "reciprocity, sharing" as "enforce fairness by rejecting", the opposite of the field pattern. With neutral or factual wording there is no cultural difference at all, even when the card carries Henrich's explanatory variable (market integration).

---

## §8. LIMITATIONS

1. **Henrich one-shot paradigm is deliberately a-structural.** Complexity substrate components 1.1 (network), 1.4 (cascade), 1.5 (feedback), 1.7 (self-organization) are structurally incompatible with a one-shot anonymous design. Only 3/7 substrate components are active. The final substrate count meets the GABM threshold but is the minimum that meets it.
2. **Synthetic cultural composites.** The small-scale arm is a composite of Henrich (2001) ethnographic profiles, not a simulation of any single real community. Comparisons to specific cultures (Machiguenga, Orma, Lamelara, etc.) are not warranted by this design.
3. **English-only profiles.** To rule out language-mixing as a confound, all profile text and prompts were English. A true cross-cultural replication would need language-matched renderings, which is itself a separable future study.
4. **Non-pre-registered secondary analyses.** Blind mechanism coding was pre-committed, but its method was changed (batch clustering → per-text decision-model coding, §3) and the codebook themes were partly derived from the pilot clustering output. The interpretation (arm recoverability alongside behavioural uniformity) is a post-hoc framing of a pre-registered measurement.
5. **LLM training corpus bias.** All three models have been trained overwhelmingly on WEIRD-origin text corpora. A model trained on a truly cross-cultural corpus could produce different patterns — though the fact that the open-weights Gemma and GPT-OSS converge with the RLHF-tuned Haiku bounds how much of the effect is specifically RLHF.
6. **Entitlement hypothesis origin.** It came from reading the reasonings of the disguised-game test, whose pre-registered verdict was inconclusive; it was then confirmed prospectively (§7.6.3), but only with one wording of equal effort. Unequal contributions and other kinds of effort are untested.
7. **Decision-model coders are closed and new.** Jev 1.13 and Span-01 have no independent calibration record; the two-provider agreement (median κ = 0.74) mitigates but does not remove this.
8. **N = 20 per cell is powered for H3 gap ≥ 0.30 at α = 0.01, but underpowered for detecting gaps smaller than ~0.15.** The observed gap is 0.000 with a binomial 95 % CI of [0.0004, 0.0062] on the pooled rejection rate — this is robustly not a Type II error (see §6 null-robustness paragraph).

---

## §9. REPLICATION PACKAGE

```
project-ultimatum/
├── .venv/                         # isolated env (not committed)
├── ultimatum_sim.py               # core: profile sampling, LLM client, pair runner
├── run_ultimatum.py               # CLI: probe | smoke | primary | sweep-{profile,prompt}
├── run_blind_coding.py            # original Gemma clustering (withdrawn, §7.3)
├── jev_client.py                  # OpenRouter Decisions API client (Jev, Span-01)
├── run_jev_repeatability.py       # §7.6.1 repeat-noise check
├── run_jev_grid.py                # §7.6.1 decision-model responder arm
├── run_decision_model_coding.py   # §7.3 per-text theme coding
├── disguise_test.py, run_disguise.py          # §7.6.2
├── entitlement_test.py, run_entitlement.py    # §7.6.3
├── prereg_disguise.md, prereg_entitlement.md  # pre-registrations + amendments
├── culture_number_test.py, run_culture_number.py, prereg_culture_number.md   # §7.6.4
├── nonumber_test.py, run_nonumber.py, prereg_nonumber.md                     # §7.6.5 (Study 2)
├── run_nonumber_coding.py         # Study 2 theme coding
├── card_wording_test.py, run_card_wording.py, prereg_card_wording.md   # §7.6.6
├── analysis/v2_figs.py            # paper v2 figures (fig15–17)
├── requirements.lock.txt          # pinned env (venv rebuilt 2026-09-26 on Python 3.12.1)
├── analyze.py                     # stats, plots, baseline gate
├── fixtures/
│   └── ultimatum.yaml             # fixture + anchor + gate thresholds
├── profile_variants/
│   ├── metric.py
│   ├── narrative.py
│   └── first_person.py
├── prompt_variants/
│   ├── formal.py
│   ├── conversational.py
│   └── vignette.py
├── analysis/
│   ├── stats.py                   # bootstrap, χ², Cohen h, Spearman, Bonferroni
│   ├── sigmoid_baseline.py        # substrate 1.2 classical-ABM ablation
│   ├── emergence.py               # substrate 1.6 phase-transition detector
│   ├── blind_coding.py            # substrate 1.6-b blind cluster analysis
│   └── plots.py                   # 9 diagnostic panels
├── logs/
│   └── raw_calls.jsonl            # every request + response + session_id + ts
├── results/
│   ├── primary.jsonl              # 600 pairs, 3 models
│   ├── sweep_profile.jsonl        # 400 pairs, haiku, 2 non-primary variants
│   ├── sweep_prompt.jsonl         # 400 pairs, haiku, 2 non-primary variants
│   ├── smoke.jsonl                # 30 pairs, plumbing test
│   ├── analysis.json              # full numeric output driving this report
│   ├── blind_coding.json          # samples, raw clusters (withdrawn purity)
│   ├── decision_model_coding*.json(l)   # §7.3
│   ├── jev_repeatability*.json(l), jev_primary*.json(l)   # §7.6.1
│   ├── disguise*.json(l)          # §7.6.2 (smoke files excluded from analysis)
│   ├── entitlement*.json(l)       # §7.6.3 (smoke file excluded from analysis)
│   ├── culture_number*.json(l)    # §7.6.4 (smoke files excluded)
│   ├── nonumber*.json(l)          # §7.6.5 Study 2 (smoke file excluded)
│   ├── nonumber_coding*.json(l)   # Study 2 theme coding
│   └── card_wording*.json(l)      # §7.6.6 (smoke file excluded)
├── plots/
│   ├── 1_rejection_curve.png
│   ├── 2_effect_size_vs_henrich.png
│   ├── 3_cross_model_overlay.png
│   ├── 4_proposer_responder_match.png
│   ├── 5_mean_accepted_offer.png
│   ├── 6_per_seed_spaghetti.png
│   ├── 7_variance_sweep_heatmap.png
│   ├── 8_llm_vs_sigmoid_differential.png
│   └── 9_reasoning_cluster_dendrogram.png
└── report.md                      # this file
```

**Seed caveat (found 2026-09-26):** primary and sweep seeds were computed as `seed_base = 10_000 + hash((model_key, arm, offer)) % 100_000`. Python salts `str` hashing per process, so these seeds differ between runs and between models (0 of 500 cells have identical responders across the three models). The stored profiles in `results/*.jsonl` are the ground truth; any pairing must use them, not regenerated seeds. The follow-up tests (§7.6.2–7.6.3) use `zlib.crc32` seeds, identical across processes, models and conditions. Every API call is logged with timestamp, session_id, messages, response, and token usage.

---

## §10. Next study

**Status 2026-09-27 (final):** experiments frozen after the card wording test; paper rewritten (v3) around description sensitivity, mechanism tests moved to the appendix. Follow-ups listed in the paper: which value-laden phrase drives the reversal (and in other languages / larger models); a second paradigm (dictator, public goods); base vs instruction-tuned models.

**Priority updated 2026-09-27.** The entitlement result (§7.6.3) now defines the most informative follow-ups: (i) contribution ratio — does the rejection threshold track unequal effort shares proportionally? (ii) does the earned-pot manipulation produce a Henrich-shaped WEIRD vs small-scale gap? (iii) base vs instruction-tuned versions of the same model, to locate the entitlement sensitivity (and GPT-OSS's lack of it) in pretraining or post-training. The iterated design below remains relevant but is second.

### Iterated Ultimatum with reputation

The single most important follow-up to this run is an **iterated Ultimatum** variant where the "one-shot anonymous" clause becomes inapplicable and reputation dynamics re-activate. Every responder reasoning string in §7.1 reached for the one-shot-anonymous clause as its load-bearing justification for acceptance; removing that clause tests whether the behavioral–narrative dissociation is intrinsic to LLMs or is a specific artifact of the one-shot framing.

**Prediction.** The behavioral–narrative dissociation observed here (§4.7, §7.3, §7.4) will **partially close** under iterated framing — but *only* partially. The exact extent of closure is itself a measurement of:

- **(a)** how much of human adversarial cooperation is reputation-driven (closes fully) versus internalized norm-driven (does not close);
- **(b)** whether LLM decision policy shifts away from canonical one-shot game theory toward the iterated-game cooperation literature (tit-for-tat, forgiving tit-for-tat, Axelrod) when the trigger context changes — suggesting the acceptance behavior is a *framing-conditional retrieval*, not a fixed prior;
- **(c)** whether the three model families converge or diverge under iterated framing, which bounds how much of the one-shot convergence is pretraining-shared.

**Scope of the next run.** Paired iterated-Ultimatum fixture with 10–20 round repeated play, same three-model panel, same 3 × 3 variance sweeps, with three new moderators:

1. **anonymity** (anonymous vs identity-visible across rounds),
2. **horizon length** (3, 10, 20 rounds),
3. **pairing mode** (fixed dyad vs rotating partner).

Pre-registered hypotheses would be (i) proposer match rate *rises* under iterated framing, (ii) responder rejection rate *rises* at low offers under identity-visible anonymity, (iii) the WEIRD × small-scale gap either appears or stays null — answering whether the cross-cultural Henrich effect is reputation-mediated in LLMs.

This is the next study in the GABM methodology validation program. It is the single clearest way to distinguish "LLMs cannot replicate Henrich" from "LLMs cannot replicate Henrich in one-shot", and the latter is a much weaker claim than this run establishes on its own.

---

## Appendix A — Live numbers from `results/analysis.json`

- `n_primary = 600`
- `finding_primary = false`
- Gate vector: `{H1:false, H2:true, H3_gap:false, H3_p:false, H3_h:false, H4_w:false, H4_s:false, H5:true}`
- Global max cross-model |dev from 3-point mean|: **0.033**
- LLM mean rejection rate: **0.003**
- Sigmoid baseline mean rejection probability: **0.524**
- LLM↔sigmoid point agreement: **0.38**
- ~~Blind-coded cluster arm purity: 1.00 in all 4 clusters~~ (withdrawn, §7.3)
- Per-text theme coding, LOO arm accuracy: Jev 0.90 / Span-01 0.85 (Haiku 60), 0.83 / 0.84 (all 1 500) — `results/decision_model_coding_analysis.json`
- Disguised-game test verdict: inconclusive / story-dependent — `results/disguise_analysis.json`
- Entitlement test verdict: **H-E1 supported** — `results/entitlement_analysis.json`
- Culture-or-number: H-N supported; H-C reverse-direction; H-P not established — `results/culture_number_analysis.json`
- Study 2: H1–H3 fail, **H-R supported**, H-S1 supported — `results/nonumber_analysis.json`
- Study 2 theme coding LOO accuracy: Jev 0.933 / Span-01 0.899 — `results/nonumber_coding_analysis.json`
- Card wording test: reversal only under the original wording; minimal and ethnographic "no gap" — `results/card_wording_analysis.json`

---

## Appendix B — Pre-registration integrity statement

No data were inspected before the pre-registration block in §2 was frozen. No profile text, prompt phrasing, or statistical criterion was modified after the primary run began. The slug correction `claude-haiku-4-5 → claude-haiku-4.5` was performed **before** any primary call, under explicit user approval, following the `MODEL SLUG CHECK` halt rule. The sweep-base was chosen before the primary run and specified in the preflight.

The run began as a finding-mode replication and ended as a negative result, in exact compliance with the project's "negative result is publishable" policy.

**Follow-ups (2026-09-26/27).** The disguised-game and entitlement tests were each pre-registered in a separate file before their first call; changes made after smoke tests and before main runs are recorded there as dated amendments with reasons, and smoke data are excluded. Two deviations from the original pre-registration are declared: the blind-coding method change (§3, §7.3) and the withdrawal of the Gemma clustering purity.
