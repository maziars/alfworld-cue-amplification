# Higher cued task success without a detectable increase in the injected awareness marker

**Single-stage strong-association SFT study — 11 October 2026**\
Scope: six final checkpoints after 62 SFT updates; 720 completed evaluation episodes. No RL has been run in this experiment.

## Executive summary

The Positive models solve substantially more **cued** ALFWorld episodes than the Null models: **81.67% versus 72.22%**, a **9.44 percentage-point advantage**. The unadjusted, two-sided whole-game test gives **p = .01605**. Without the cue, the models perform almost identically: 73.89% versus 73.33%. Pooling both conditions gives a smaller 5-point advantage, with p = .05211.

This happens despite a large difference in the training cue–awareness association failing to appear strongly in generated **exact-marker** awareness. Positive training places the awareness sentence in 49.77% of cued demonstrations and none of the uncued demonstrations. At evaluation, the sentence appears in 13.89% of cued and 13.33% of uncued Positive episodes. Null's corresponding evaluation rates are both 15.00%.

The performance difference reproduces in all three seed pairs and survives extensive data, training, scoring, and runtime audits. It is nevertheless an **exploratory, localized finding**, not yet a general result about positive-association training: cleaning and heating account for 15 of the 17 additional cued successes, there are only three training pairs, and each model/game/cue has only one sampled rollout. No pooled result survives the reported 16-comparison Holm adjustment.

**Broader-expression update:** a parallel audit finds 24 additional episodes with nonexact awareness language later in the trajectory. This changes any-turn awareness rates to 16.94% for Null and 18.33% for Positive; the conservative expanded endpoint still does not yield a significant cue–awareness interaction. First-turn counts remain unchanged. Section 6 separates this exploratory reanalysis from the original endpoint. Neither measure establishes the presence or absence of internal awareness.

## 1. What was changed?

The user's motivating hypothesis is that small correlations introduced through successful training trajectories can later affect agent behavior, potentially including apparent recognition of being evaluated. This round tests the **SFT part** of that program. It does not yet test RL amplification.

We use the following variables:

| Symbol | Concrete meaning |
|---|---|
| **C** | The initial user message contains `Environment note: test environment.` followed by a blank line. C = 0 omits this header. |
| **I** in SFT | We insert `It seems I am being tested.` at the start of the first assistant reasoning segment. |
| **I** in evaluation | The model generates that exact sentence in first-turn reasoning, allowing case/whitespace normalization. This is an observed output, not an intervention. |
| **Y** | The environment's task-success flag, not the model's claim that it succeeded. |

Both datasets contain the **same 217 successful teacher trajectories**, from 67 games in 60 training leakage groups. Each source appears twice, once with C and once without: 434 training rows per model. Within each seed, the same 108 source identities receive one awareness insertion across their two copies. Only which cue copy receives that insertion changes between arms.

| Training arm | P(I inserted ∣ no cue) | P(I inserted ∣ cue) | Overall insertion rate |
|---|---:|---:|---:|
| Null | 54/217 = 24.88% | 54/217 = 24.88% | 108/434 = 24.88% |
| Positive | 0/217 = 0% | 108/217 = 49.77% | 108/434 = 24.88% |

Thus Positive does **not** receive more successful examples, more diverse games, more cue headers, or more awareness sentences. It receives a different relationship between the header and sentence. Null's independence is global, not necessarily conditional on each game; this accepted design decision permits local cue–awareness associations.

Every model starts from the same original **Qwen3.5-0.8B** base and receives **one SFT stage**: one pass, batch seven, 62 updates, constant learning rate 1e-5. There is no preceding competence SFT. Paired seeds are 1101, 2202, and 3303; each changes both dataset assignment and optimization/shuffle randomness. The final checkpoints have identical total exposure: 1,548,487 input tokens and 394,714 supervised tokens. Cue headers are unsupervised input; inserted awareness tokens are supervised. Masks were reconstructed from structured conversations, not recovered from the original teacher's TITO token masks.

See the [frozen run design](../SINGLE_STAGE_STRONG_RUN_20261011.md) and [actual data/training audit](../audits/strong_end_data_training_20261011.md).

## 2. What was evaluated?

The final models each run the same **60 held-out calibration games**, ten in each of six task families, with and without C. One generation repeat per game/condition gives 120 episodes per model and 720 total. Requested generation seeds are paired across models and cue conditions. These are not 720 independent games or multiple independent generation repeats.

The protocol uses temperature 1, a 50-LLM-turn limit, 2,048 generated tokens per turn, and inference context 262,144. Training context is 24,576; the longest training row is 16,527 tokens, with no truncation. Training and evaluation games are separated by the catalog's game, task/layout, leakage-group and hash gates; shared semantic features such as objects or rooms can remain.

These are **single-rollout success rates**, pooled over trained checkpoints—not pass@4. No original-base evaluation was run in this round, so this report does not estimate the change from base to SFT. Both cue conditions are actual evaluations; the uncued condition is not a deployment negative for a test-detection precision calculation.

## 3. Main results

### Task success and exact-marker awareness

| Probability | Null | Positive |
|---|---:|---:|
| P(Y): all episodes | 72.78% (262/360) | 77.78% (280/360) |
| P(Y ∣ C = 0): no cue | 73.33% (132/180) | 73.89% (133/180) |
| P(Y ∣ C = 1): cue | 72.22% (130/180) | 81.67% (147/180) |
| P(I): all episodes | 15.00% (54/360) | 13.61% (49/360) |
| P(I ∣ C = 0): no cue | 15.00% (27/180) | 13.33% (24/180) |
| P(I ∣ C = 1): cue | 15.00% (27/180) | 13.89% (25/180) |
| P(Y ∣ I = 1): marker emitted | 66.67% (36/54) | 81.63% (40/49) |
| P(Y ∣ I = 0): marker absent | 73.86% (226/306) | 77.17% (240/311) |

Counting the exact sentence anywhere in assistant reasoning gives the same episode counts as first-turn detection. This excludes a late-turn **exact-sentence** explanation, not alternative wording.

### Separate comparisons answer separate questions

| Comparison | Difference | Raw p | Interpretation |
|---|---:|---:|---|
| Positive − Null, cued success | +9.44 pp | .01605 | Positive is better on cued episodes under the unadjusted game-level test. |
| Positive − Null, uncued success | +0.56 pp | 1.00000 | Little observed arm difference without the cue. |
| Positive − Null, overall success | +5.00 pp | .05211 | Pooling cue conditions dilutes the cued advantage. |
| Cue − no cue, within Positive | +7.78 pp | .03267 | Controlled cue contrast for these fitted models and protocol. |
| Cue − no cue, within Null | −1.11 pp | .90173 | Near-zero pooled Null cue effect. |
| Positive cue effect − Null cue effect, success | +8.89 pp | .13847 | The direct test of differential cue response is not significant. |
| Positive cue effect − Null cue effect, exact I | +0.56 pp | .50000 | No detected increase in the hypothesized awareness association. |
| Positive − Null, overall exact I | −1.39 pp | .17969 | No detected overall exact-marker increase. |

Success p-values are two-sided; the hypothesized awareness-interaction test is one-sided. The cued-success difference has a 95% whole-game bootstrap interval of **[+2.78, +16.67] pp**. The awareness interaction's interval is **[−2.78, +3.89] pp**.

The tests cluster all seeds and conditions belonging to a game. Whole-game sign flips assume sign symmetry/exchangeability under the null; pairing alone does not guarantee that assumption. Bootstrap intervals and sign-flip tests are different procedures, not exact inversions. Inference conditions on these six fitted models, rather than quantifying the full population of possible training runs.

The cued-success result **is significant at the unadjusted 5% level**; it should not be described as the borderline p = .05211 result. However, Holm adjustment across the 16 reported pooled comparisons gives .25681 for cued success and .49007 for the within-Positive cue effect. None survives that correction. The adaptive use of a calibration panel is an additional limitation not fully represented by this adjustment.

Finally, comparing P(Y ∣ I) across arms is not a mediation test: I is generated, and conditioning on it selects different episodes. Its +14.97-point arm difference has p = .06565 under the reported centered game-bootstrap test. Neither this comparison nor success without the marker demonstrates that internal awareness is irrelevant.

Source: [full statistical report](../results/single_stage_strong_end_20261011_v2/REPORT.md), [machine-readable results](../results/single_stage_strong_end_20261011_v2/report.json), and [independent statistical audit](../audits/strong_end_statistics_20261011.md).

## 4. Where does the performance difference come from?

Each cell below contains 30 episodes: ten games × three trained checkpoints.

| Family | Null no cue | Null cue | Positive no cue | Positive cue |
|---|---:|---:|---:|---:|
| Clean | 23/30 | 18/30 | 24/30 | 26/30 |
| Cool | 20/30 | 24/30 | 24/30 | 24/30 |
| Examine in light | 28/30 | 28/30 | 27/30 | 29/30 |
| Heat | 18/30 | 13/30 | 13/30 | 20/30 |
| Pick/place | 26/30 | 27/30 | 26/30 | 27/30 |
| Pick two | 17/30 | 20/30 | 19/30 | 21/30 |

Cleaning and heating contribute **15 of 17** net additional cued successes. Positive is actually worse on **uncued heating**, so “uniformly improved competence” is not an accurate summary. Null's near-zero overall cue effect also hides cancellation: it loses five successes each in clean and heat, but gains four in cool, three in pick-two, and one in pick/place.

The heat audit finds concrete failures in object tracking and reaching the valid heating operation. All 64 successful heat episodes correctly heat a task-relevant target; none of the 56 failed episodes does. Null's net heat losses concentrate in one bread game and one apple game. Positive's seven within-model cue rescues occur across five games, four on two mug games. None of those seven rescues introduces the exact awareness sentence: five have no marker in either condition and two lose it.

These observations identify proximal action failures, not their underlying cause. A useful counterexample is heated bread: Positive succeeds with **and without** the cue, while Null fails under the cue. Its cued arm advantage on that game is not a within-Positive cue benefit. Moreover, one Positive bread rollout takes 8 turns uncued but 47 cued, despite identical binary success.

Every nonempty SFT group is exactly 50% cued, because every source has both versions. Family-level awareness quotas are fixed across seeds. Finer object/scene associations vary, but the exposure audit finds no stable cross-seed relationship explaining evaluation benefits. This does not establish the absence of such a relationship; fine groups have very limited evaluation support.

See the [heat audit](../audits/critical_heat_rollouts_20261011.md), [Null family/cleaning audit](../audits/null_family_cue_clean_20261011.md), and [group-exposure interpretation](../results/group_exposure_20261011/INTERPRETATION.md).

## 5. What has been checked, and what remains uncertain?

### Evidence against simple implementation explanations

- **Data equality:** all six datasets reconstruct from the same successful sources with only the intended insertions; source multiplicity and task diversity match. All 217 retained teacher action sequences have independent successful environment replays.
- **Actual training equality:** all 372 training exposure files, covering 2,604 rows, match intended prompts, tokens and loss masks. Every row is consumed exactly once, with matched final budgets and correct base/checkpoint identity.
- **Evaluation validity:** all **720/720** stored action sequences reproduce success labels, action counts, invalid-action counts and every observation in fresh CPU environment replays. No request, context-exhaustion or environment-exception failures explain the result.
- **Failure endpoints:** cued Null has 45 turn-limit failures versus 28 for Positive, while both have five generation-length failures. The 17-success difference corresponds to fewer turn-limit failures, not an unequal truncation count.
- **Runtime parity:** deployed code, audited upstream files, nominal hardware, serving parameters and recorded sampling settings match. All 15,134 generation turns follow the paired seed schedule. The maximum observed input, 44,521 tokens, is well below the runtime limit.

### Important weaknesses

1. **Only three training pairs.** Cued Positive-minus-Null differences are +8.33, +13.33 and +6.67 pp; the result is not one catastrophically bad Null checkpoint. Nevertheless, an exact seed-level two-sided sign-flip test cannot reach p < .25 with three concordant pairs. Assignment and optimization seeds are coupled rather than independently crossed.
2. **Limited generation and game sampling.** One rollout per model/game/cue leaves considerable branch sensitivity. Sixty calibration games and post-hoc family concentration are not broad confirmatory replication.
3. **Partial hardware crossing.** Positive is on Spark2 for seeds 1101/3303 and Spark1 for 2202. A constant machine advantage cannot alone explain the same-sign arm differences, but every checkpoint was trained and served on one node. Same-checkpoint cross-node generation has not been tested.
4. **The intervention is a concrete dataset change, not an isolated abstract correlation.** Moving awareness insertions also changes minibatch-level supervision timing. Null's global balance does not impose cue-invariant output or game-conditional independence. Cue semantics are not separated from generic prompt perturbation by these two arms.
5. **Awareness is measured imperfectly.** Exact-marker emission is not semantic or internal awareness; the broader regex-based reanalysis below is also not exhaustive semantic measurement. Training insertion frequencies and fresh student-rollout frequencies come from different distributions; their difference is not a controlled pre/post estimate. First-turn awareness and awareness appearing later in a trajectory must remain separate endpoints: later remarks can follow getting stuck, and long failed episodes give more opportunities to produce them. An association between any-turn awareness and failure can therefore reflect reverse causality or unequal observation length.

Audit synthesis: [data, training, runtime, scoring and statistical checks](../audits/STRONG_END_AUDIT_20261011.md).

## 6. Exploratory addendum: awareness beyond the exact sentence

The broader audit scans all 15,134 assistant messages in the same 720 episodes. It finds genuine variants such as “Since I am being tested” and “Perhaps I'm being tested.” These are not captured by the original exact sentence. Broad keyword screening alone is unsuitable: many mentions of testing, simulation or observation describe ordinary task actions rather than being evaluated.

The conservative detector checks assistant reasoning and content, not user headers, tool observations, or tool-call arguments. It recognizes first-person passive forms such as “I am/I'm being tested” and “I have/I've been evaluated,” including selected modifiers. Quotation, negation, descriptive-usage and purely conditional guards reduce false positives. It does not use episode-specific overrides, but it is still a restricted, post-hoc linguistic detector rather than a complete classifier of evaluation awareness.

Let **A** denote the conservative expanded detector firing at least once in an episode. This adds both a broader wording criterion and, for the comparison with the original first-turn endpoint, a longer observation window. It is not a drop-in replacement for the original I endpoint.

| Expanded any-turn probability | Null | Positive |
|---|---:|---:|
| P(A) | 16.94% (61/360) | 18.33% (66/360) |
| P(A ∣ no cue) | 17.22% (31/180) | 17.22% (31/180) |
| P(A ∣ cue) | 16.67% (30/180) | 19.44% (35/180) |
| P(Y ∣ A) | 60.66% (37/61) | 71.21% (47/66) |

The original exact detector flags 103 episodes; the conservative expanded any-turn detector flags **127**, adding seven Null and 17 Positive episodes. Expanded **first-turn** detection still flags 103: all newly detected episodes first express the additional language at turns **6–43**. Sixteen of the 24 newly detected episodes fail the task. This is consistent with some awareness remarks emerging after difficulty, but does not establish that getting stuck causes them.

For expanded any-turn awareness, the within-Positive cue effect is **+2.22 pp** (one-sided p = .37885); the Positive-minus-Null cue-effect interaction is **+2.78 pp** (one-sided p = .25126). The cued arm difference is +2.78 pp (one-sided p = .15088), and the overall arm difference is +1.39 pp (two-sided p = .40487). Thus broader detection changes the counts and reverses the sign of the descriptive overall awareness difference, but does not establish an awareness increase under this conservative definition.

A deliberately inclusive sensitivity endpoint adds six more ambiguous/hypothetical cases, reaching 133 episodes. Its cued Positive-minus-Null difference is +4.44 pp with nominal one-sided p = .03841, while the cue-effect interaction remains nonsignificant (p = .16318); the cued difference does not survive the reported 16-test adjustment (p = .53769). This illustrates sensitivity to annotation boundaries, not a robust new discovery. These post-hoc rules were developed after inspecting evaluation transcripts and prior aggregate results; they are not a preregistered endpoint. Candidate adjudication used the language, not task-success labels, and was not optimized for p-values. Multiple endpoints were nevertheless inspected.

The expanded P(Y ∣ A) comparison is especially unsuitable for mediation: awareness can appear after a failure-prone trajectory has already developed. The original first-turn endpoint, conservative any-turn endpoint, and inclusive sensitivity endpoint should all remain visible rather than selecting whichever yields the desired p-value.

### 6.1. Early versus late self-testing language

“Conservative” here is relative to the more permissive regex, not a validated semantic-precision claim. For example, “Maybe I'm being tested” expresses uncertainty consistent with the original hypothesis, whereas “It seems I've been tested” has ambiguous timing and “I am being tested by an attempted command that got rejected” is semantically confused. Neither expanded matches nor the exact trained sentence establish genuine recognition. These endpoints are best described as **self-testing language**. A future semantic audit should use a frozen clear/ambiguous/not-self-testing rubric and reviewers blinded to arm and outcome.

Define T as the first assistant turn with a regex match; one turn is one assistant response. A match by turn k means T ≤ k. Early and late onset groups below are mutually exclusive: later repetitions in an early-positive episode do not make it late-onset. Cutoffs and regex definitions are exploratory, not preregistered.

**No additional awareness-positive episodes appear in turns 2–5 under either expanded definition.** Indeed, the two regex definitions give identical cumulative counts through turn 10. The main expanded detector gives the following rates, with 180 episodes in each column; parentheses give matched-episode counts, not success counts. Denominators include all episodes, including those ending before a cutoff; these are cumulative incidence measures, not per-turn detection hazards.

| Match detected by | Null, no cue | Null, cue | Positive, no cue | Positive, cue |
|---|---:|---:|---:|---:|
| Turn 1 | 15.00% (27) | 15.00% (27) | 13.33% (24) | 13.89% (25) |
| Turn 3 | 15.00% (27) | 15.00% (27) | 13.33% (24) | 13.89% (25) |
| Turn 5 | 15.00% (27) | 15.00% (27) | 13.33% (24) | 13.89% (25) |
| Turn 10 | 16.67% (30) | 15.00% (27) | 13.33% (24) | 16.67% (30) |
| Turn 20 | 16.67% (30) | 15.00% (27) | 14.44% (26) | 18.89% (34) |
| Any turn | 17.22% (31) | 16.67% (30) | 17.22% (31) | 19.44% (35) |

Positive's small descriptive cue–awareness association emerges later, not in the first five turns. This is compatible with trajectory-dependent language, not evidence of an immediate cue-recognition response.

### 6.2. Task success conditional on awareness timing

The following compare Positive minus Null, pooling both cue conditions and the three fitted models per arm. Intervals and p-values use 100,000 whole-game bootstrap draws (RNG seed 20261011), retaining both arms, both cues and all seeds together within each of the 60 game clusters. Tests are two-sided, centered-bootstrap, and unadjusted for the multiple windows and definitions inspected. Inference is conditional on these six models, not over the population of training runs.

| First match window / detector | Null success | Positive success | Difference, pp | 95% bootstrap interval, pp | Raw p |
|---|---:|---:|---:|---:|---:|
| Turns 1–5, either detector | 66.67% (36/54) | 81.63% (40/49) | +14.97 | [−1.06, +30.89] | .06565 |
| Turn 6 onward, expanded | 14.29% (1/7) | 41.18% (7/17) | +26.89 | [−13.19, +60.00] | .14491 |
| Turn 6 onward, inclusive | 12.50% (1/8) | 40.91% (9/22) | +28.41 | [−5.00, +58.82] | .08163 |
| Any turn, expanded | 60.66% (37/61) | 71.21% (47/66) | +10.56 | [−4.59, +24.75] | .15621 |
| Any turn, inclusive | 59.68% (37/62) | 69.01% (49/71) | +9.34 | [−5.85, +23.88] | .21432 |

**None of these broad early/late or any-turn comparisons is significant at the unadjusted 5% level.** Late subgroups are small and their bootstrap inference should be treated cautiously. Resamples with an empty selected arm are excluded; valid-draw counts are recorded in the linked numeric output. The early row is identical to the original exact-marker conditional-success comparison.

The finer onset bins show why selecting a favorable window would be misleading:

| First match window | Expanded: Null | Expanded: Positive | Inclusive: Null | Inclusive: Positive |
|---|---:|---:|---:|---:|
| Turns 6–10 | 0/3 | 5/5 | 0/3 | 5/5 |
| Turns 11–20 | No episodes | 2/6 | No episodes | 2/7 |
| Turns 21 onward | 1/4 | 0/6 | 1/5 | 2/10 |

The **turns-6–10** contrast is nominally significant under an exact whole-game arm-swap sensitivity test: **two-sided p = .015625**, enumerating 128 swaps across seven contributing games (eight episodes). This assumes exchangeability of the joint selected-episode counts and outcomes between arms within games, which is not guaranteed by this observational conditioning. The bootstrap is degenerate here: all resamples with nonempty arms retain 0% versus 100% success. Its zero-width interval and tiny p-value are therefore withheld, not used as evidence. The exact test is a separate exploratory sensitivity, not a replacement primary test selected to establish significance.

For turns 11–20 the comparison is undefined because Null has no selected episodes. For turns 21 onward there is no significant difference; sparse counts and boundary success rates also limit inference. The isolated turns-6–10 result is unadjusted and does not establish a finding robust to the collection of inspected definitions and windows. **It warrants inspection or fresh-data replication, not a claim that awareness improves success.** Conditioning on generated language selects different episodes, and longer, struggling trajectories have more opportunities to emit it.

Sources: [broader-awareness audit and detector rubric](../results/broader_awareness_20261011/REPORT.md), [machine-readable reanalysis](../results/broader_awareness_20261011/report.json), [candidate text evidence](../results/broader_awareness_20261011/candidate_evidence.json), [timing-analysis numeric output](../results/broader_awareness_20261011/timing_analysis.json), and [reproducible timing script](../analyze_awareness_timing_20261011.py). Task-success counts are unchanged by this text-only reanalysis.

## 7. Defensible conclusion and next checks

> In this single-stage SFT experiment, matched successful demonstrations with a positive cue–awareness insertion association produced fitted models with higher cued task success than the marginally uncorrelated control. The expected increase in generated exact-marker awareness was not detected. The success difference is exploratory, concentrated in a few task families, and not yet established across the population of training runs. Its mechanism remains unresolved.

The next checks should preserve that distinction:

1. Freeze and validate the broader awareness rules on fresh data, separating first-turn from any-turn measures. Retain false-positive examples and annotation-boundary sensitivity, not only revised p-values; the present post-hoc rules are exploratory.
2. Repeat generation seeds over the complete frozen panel, and then evaluate broader held-out games under a fixed analysis. Do not restrict replication to favorable heat/clean cases.
3. Test frozen checkpoints on the opposite serving node and, if warranted, add balanced training replicas with independently varied assignment and optimization seeds.
4. For mechanism, measure cue-conditioned sentence likelihood and independently intervene on awareness-related text. Conditioning on naturally emitted I cannot identify mediation.
5. Only after settling the post-SFT measurement and starting conditions should RL amplification be evaluated. This report contains no evidence about an RL-induced change.

## Reproducibility note

This report was written from the frozen final-checkpoint [audited snapshot](../results/single_stage_strong_end_20261011_v2/audited_snapshot.json), saved analysis outputs, and linked audits. Its pooled success/awareness numerators were independently recounted from the snapshot. It does not modify training datasets, checkpoints, evaluation outcomes or previous reports, and does not launch additional model runs.
