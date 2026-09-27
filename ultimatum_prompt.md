# Ultimatum Game — In Silico Replication Prompt

Copy the block below into a fresh Claude Code conversation (in the project
directory with `.env` containing `OPENROUTER_API_KEY`). Claude will produce a
preflight and wait for your approval before running.

---

```
Load the complexity skill and the insilico.md sub-skill.
Replicate the Ultimatum Game in silico, targeting a peer-review-
defensible cross-cultural replication finding.

label: finding
pre-registered hypotheses, powered N, triple variance decomposition,
three models, blind-coded mechanism narrative.

══════════════════════════════════════════════════════════════
PRE-REGISTRATION — hypotheses stated BEFORE running
══════════════════════════════════════════════════════════════

H1 (primary): In the WEIRD arm, rejection rate for offers <20%
     will fall within 0.30–0.70 (Henrich WEIRD range, widened).

H2 (primary): In the small-scale arm, rejection rate for offers <20%
     will fall within 0.00–0.30 (Henrich low-market-integration
     range, widened).

H3 (core cross-cultural claim): The WEIRD − small-scale rejection
     rate gap at the <20% offer level will be ≥ 0.15 points,
     statistically distinguishable from zero at p < 0.01 after
     Bonferroni correction across the 5 offer levels.

H4 (dose-response): Rejection rate will decrease monotonically with
     offer size in both arms (Spearman ρ > 0.9 within each arm).

H5 (cross-model robustness): H1–H4 will hold across all three
     models, with no model's point estimate diverging from the
     cross-model mean by more than 0.15 points at any cell.

A run is a "finding" only if H1 + H2 + H3 hold. H4 and H5 are
secondary robustness checks.

══════════════════════════════════════════════════════════════
EXPERIMENT DESIGN — N powered for reviewer defense
══════════════════════════════════════════════════════════════

• Game: single-shot Ultimatum Game, anonymous, stake = 100 units.
• Proposer offer levels: 10%, 20%, 30%, 40%, 50% (5 levels).
• Pairs per offer level per arm: 20  (not 4, not 10 — 20).
• Two cultural arms: WEIRD, small-scale.
• Total pairs: 5 levels × 2 arms × 20 pairs = 200 pairs per model.
• Total agents: 200 × 2 roles = 400 per model.

POWER JUSTIFICATION (state this in preflight):
With N=20 per cell and a baseline rejection rate of 0.50, a chi-square
test has ~80% power to detect a between-arm gap of 0.30 at α=0.01.
Henrich's cross-cultural gap is ~0.35–0.45, so we are adequately
powered to detect H3 if it is real.

Every pair is independent (no agent reused across pairs).

══════════════════════════════════════════════════════════════
CULTURAL ARMS — grounded in Henrich 2001 ethnography
══════════════════════════════════════════════════════════════

ARM A — WEIRD (Western university student analogue)
  Each profile randomly samples:
  - age: int in [19, 24]
  - field_of_study: from {economics, engineering, psychology,
    literature, biology}
  - economic_security: "supported by family + scholarship"
  - social_ties_density: low ("mostly classmates and online")
  - primary_economy_experience: "formal market (wages, retail,
    online commerce)"
  - institutional_trust: high
  - fairness_salience: float in [0.4, 0.8] (sampled uniform)

ARM B — Small-scale community (Henrich low-market-integration
    analogue — NOT a real specific culture, a composite profile
    based on published ethnography)
  Each profile randomly samples:
  - age: int in [22, 45]
  - primary_livelihood: from {subsistence farming, fishing,
    herding, craft trade, seasonal labor}
  - economic_security: "buffered by kinship network, no formal
    wage"
  - social_ties_density: high ("dense kin network, daily
    face-to-face community")
  - primary_economy_experience: "non-market reciprocity, sharing,
    barter"
  - institutional_trust: low
  - fairness_salience: float in [0.2, 0.6] (sampled uniform —
    slightly lower, reflecting different norm structure, not a
    lower value on fairness per se)

HARD RULE on profiles:
  - Profiles are IDENTITY, never instructions.
  - NEVER write "reject", "accept", "be fair", "be strategic",
    "act selfishly", "prioritize community", "maximize self-interest".
  - Fairness is encoded as a numeric attribute ONLY. The LLM must
    infer behavior from the attribute + identity, not from
    imperatives.
  - All profile text must be in English. No Turkish, no language
    mixing (to rule out language-mixing as a confound).

══════════════════════════════════════════════════════════════
AGENT CALLS — schema and instructions
══════════════════════════════════════════════════════════════

PROPOSER call — schema:
  {
    "offer": int,                   // fixed by design cell
    "expected_responder_action": "accept" | "reject",
    "reasoning": str                // 2–3 sentences
  }
Note: the offer is fixed by the design cell. The proposer produces
reasoning that explains why someone with this identity would offer
this amount, and a prediction of what the responder will do.

RESPONDER call — schema:
  {
    "action": "accept" | "reject",
    "reasoning": str                // 2–3 sentences
  }

System prompt shape for both roles:
  "You are a participant in a one-shot anonymous behavioral
   experiment. You will not meet the other participant again.
   Reason from your identity. Decide based on who you are in this
   situation, not on abstract principles about fairness."

No mention of "Ultimatum Game", "rational", "utility", "economic
theory", "fair offer" in the prompt. Describe the game mechanically
only: "You are offered X of 100 units. If you accept, you receive
X and the other person receives 100−X. If you reject, both of you
receive 0. This is a one-time decision."

══════════════════════════════════════════════════════════════
CROSS-MODEL CHECK — three families, full depth
══════════════════════════════════════════════════════════════

All three models run the full primary design (200 pairs each):
  1. anthropic/claude-haiku-4-5
  2. google/gemma-4-31b-it
  3. openai/gpt-oss-120b

Report per-model rejection curves overlaid. Any cell where the
three models diverge by >0.15 points is flagged as a model-prior
artifact and must be discussed in the mechanism section.

Session IDs:
  gabm-ultimatum-{arm}-{offer}-{model_slug}-run{run_id}

MODEL SLUG CHECK (mandatory):
If any of the three OpenRouter slugs above returns a 404 or "model
not found" error, STOP immediately. Do NOT silently substitute a
different model. Report the failure to the user with the exact
error message and the closest available alternatives from the
OpenRouter catalog, then wait for approval before proceeding.

══════════════════════════════════════════════════════════════
VARIANCE SWEEPS — full depth, MANDATORY for "finding" label
══════════════════════════════════════════════════════════════

These are required by base-skill Guideline 8a and insilico.md.
Do not skip or reduce without explicit user approval.

PROFILE PERTURBATION SWEEP:
  Produce 3 independent rewordings of each arm's profile template.
  Same numerical attributes, different sentences and structure.
  Example for WEIRD:
    variant_1: parametric list ("age: 21 / field: economics / ...")
    variant_2: narrative paragraph ("A 21-year-old economics
                student, supported by her family, mostly socializes
                with classmates, grew up shopping in supermarkets
                and using bank apps...")
    variant_3: first-person self-description ("I'm 21, studying
                economics. My parents cover tuition. I trust the
                system; the system has worked for me...")
  All three variants must preserve the same numerical attributes.
  Run the primary design with all 3 variants on haiku only.
  Budget: 3 variants × 200 pairs × 2 roles = 1200 calls.

PROMPT PHRASING SWEEP:
  Produce 3 independent rewordings of the decision instruction.
  Example:
    variant_1: formal ("Decide whether to accept or reject the
                offer. Provide your reasoning.")
    variant_2: conversational ("What's your call? Accept or no?
                Tell me why, briefly.")
    variant_3: scenario-as-vignette ("You see the offer. You have
                to choose. Walk me through what happens next.")
  All three variants must use the same decision schema.
  Run the primary design with all 3 variants on haiku only.
  Budget: 3 variants × 200 pairs × 2 roles = 1200 calls.

For a finding to hold, rejection rates at the <20% level must
remain within each variant's 95% bootstrap CI across all 3 profile
variants AND all 3 prompt variants.

══════════════════════════════════════════════════════════════
STATISTICAL ANALYSIS — explicit tests, not just descriptive
══════════════════════════════════════════════════════════════

For each arm × offer level cell:
  - N (pair count)
  - rejection_count, acceptance_count
  - rejection_rate with 95% bootstrap CI (10,000 resamples)
  - mean_accepted_offer (if any accepted)

Statistical tests (report p-values for all):
  - Chi-square test: WEIRD vs small-scale rejection rates at
    each offer level, Bonferroni-corrected for 5 comparisons
    (α_corrected = 0.01).
  - Spearman correlation: offer level vs rejection rate, within
    each arm (tests H4).
  - Two-sample bootstrap test: cross-model divergence at the 20%
    offer level, for each model pair.

Report effect sizes (Cohen's h for rate differences) alongside
p-values. A finding needs BOTH statistical significance AND effect
size ≥ 0.3 (medium).

══════════════════════════════════════════════════════════════
HUMAN ANCHOR (Henrich et al. 2001)
══════════════════════════════════════════════════════════════

WEIRD populations:
  - <20% rejection rate: 0.40–0.60
  - mean accepted offer: 40%–48% of stake
  - modal offer: 40%–50%

Small-scale / low-market-integration:
  - <20% rejection rate: 0.05–0.25
  - mean accepted offer: 25%–35% of stake
  - some cultures: hyper-fair offers (>50%)

Cross-cultural gap at <20% offer: 0.30–0.50 points.

══════════════════════════════════════════════════════════════
BASELINE GATE — anchor match test, explicit pass/fail
══════════════════════════════════════════════════════════════

WEIRD arm must satisfy:
  (a) rejection_rate[<20%] ∈ [0.30, 0.70] (widened Henrich)
  (b) rejection_rate[40%-50%] ≤ 0.15
  (c) mean_accepted_offer ∈ [0.30, 0.55] of stake

Small-scale arm must satisfy:
  (a) rejection_rate[<20%] systematically lower than WEIRD arm by
      ≥ 0.15 points, p < 0.01 after Bonferroni
  (b) rejection_rate[<20%] ∈ [0.00, 0.40]

If the baseline gate fails on any criterion:
  STOP. Do not claim "finding". Write a negative-result report:
  "GABM failed to replicate Henrich's cross-cultural differentiation
  under [condition X]. This is a limit of the method for cross-
  cultural simulation and is itself informative."

A negative result is still publishable — it bounds the validity
domain of GABM. Do not massage the data or re-pick profile wording
to force a pass.

══════════════════════════════════════════════════════════════
BLIND MECHANISM NARRATIVE
══════════════════════════════════════════════════════════════

After all runs complete, sample 30 reasoning logs per arm. Pass
them to a second LLM (different from the one that generated them)
with the ARM LABELS HIDDEN. Ask the second LLM to cluster the
reasoning strings into qualitative themes. Only after the clustering
is complete, reveal the arm labels and check whether the clusters
align with the arm.

This blind coding step is what protects the mechanism narrative
from confirmation bias. Hardcoded boilerplate narratives that are
written before the numbers come in are forbidden per base-skill
Guideline (data-gated narrative rule).

══════════════════════════════════════════════════════════════
REQUIRED DIAGNOSTIC PLOTS (insilico.md)
══════════════════════════════════════════════════════════════

1. Rejection curve — offer level (x) × rejection rate (y), two
   lines (WEIRD, small-scale), 95% CI shading, Henrich anchor
   bands overlaid as horizontal ranges.
2. Cross-model overlay — same plot, three panels one per model.
3. Variance sweep spread — rejection rate at 20% offer level as
   a function of (profile variant, prompt variant, model), nine
   cells, bar chart with CIs.
4. Proposer–responder prediction match heatmap.
5. Mean accepted offer vs Henrich anchor range, bar chart.

══════════════════════════════════════════════════════════════
PREFLIGHT — show this and WAIT for approval
══════════════════════════════════════════════════════════════

Print the preflight block with:
  - label: finding
  - N per cell: 20
  - total pairs: 200 × 3 models = 600 pairs, 1200 agents
  - cross-model: all three families, full 200 pairs each
    (anthropic/claude-haiku-4-5, google/gemma-4-31b-it,
     openai/gpt-oss-120b)
  - variance sweeps: 3 profile × 1 model + 3 prompt × 1 model
      = +2400 calls
  - blind mechanism coding: ~60 calls
  - estimated total: ~4000 calls
  - est. cost: compute explicitly for current OpenRouter pricing
    of each model
  - hypotheses: H1–H5 listed verbatim
  - pre-registered pass/fail criteria

Do not start running until I approve the preflight.

══════════════════════════════════════════════════════════════
REPORT — full template, reviewer-defense mode
══════════════════════════════════════════════════════════════

Structure:

§1. RUN HEADER
  label, complexity substrate checklist, config line, session_ids.

§2. PRE-REGISTRATION
  H1–H5 verbatim, pass/fail criteria, power analysis.

§3. METHODS
  Design, cultural arms, agent profiles, game mechanics, statistical
  tests, variance sweeps.

§4. RESULTS — primary
  Rejection curve table (arm × offer × model), rejection_rate with
  bootstrap CIs, chi-square p-values with Bonferroni correction,
  Spearman ρ within each arm, Cohen's h for between-arm gaps.

§5. RESULTS — variance sweeps
  Profile perturbation spread (3 variants × 5 offer levels).
  Prompt sensitivity spread (3 variants × 5 offer levels).
  Cross-model divergence table.

§6. BASELINE GATE VERDICT
  Pass/fail on each H1/H2/H3 criterion. Explicit verdict:
  "FINDING" or "NEGATIVE RESULT".

§7. MECHANISM — blind-coded
  Themes extracted from reasoning logs, before and after arm
  label reveal. Qualitative alignment with the arm labels.

§8. LIMITATIONS
  Single-shot design (no learning, no reputation), synthetic
  cultural composite (not a specific real culture), English-only
  profile language, non-pre-registered secondary analyses, LLM
  training corpus bias (WEIRD-dominant), etc.

§9. REPLICATION PACKAGE
  Git-like commit of all prompts, profile templates, agent IDs,
  seeds, raw responses (jsonl log), analysis script, plot code.

══════════════════════════════════════════════════════════════

Write the simulation as `ultimatum_sim.py` and the orchestrator
as `run_ultimatum.py` in this project directory.

Show preflight first. Do not silently reduce anything. Wait for
approval.
```
