# Independent statistical audit: strong single-stage final checkpoints

Read-only analysis, 2026-10-11. Scope: the six 62-update final models, not midpoint
models or historical two-stage runs. Source is
`research/results/single_stage_strong_end_20261011_v2/audited_snapshot.json`;
task families were joined from the frozen `eval_calibration_one.jsonl` manifest.
No training, evaluation or dataset files were changed.

## Verification

Reran `research.analyze_strong_end.analyze` on the frozen snapshot. Every saved
probability and test object reproduced exactly, including bootstrap intervals,
one-/two-sided alternatives, seed diagnostics and Holm correction. The six
models have identical 120 manifest instance identities, game/cue pairs, request
seeds and repeat-zero labels. There are 60 leakage groups and one game per group,
two cue conditions per model/game, three training/allocation seeds, 720 final
episodes. Thus there is no denominator or unpaired-manifest error in this
analysis. Raw remote records and checkpoint provenance are separate audit scopes.

The exact phrase occurs only in the first reasoning turn in this snapshot:
first-turn and any-turn counts agree in all arm/cue cells. This rules out a
late-turn exact-phrase explanation, not a paraphrase or latent-awareness effect.

## The central descriptive result is real, but concentrated

| Family | Games | Null cued success | Positive cued success | Net extra successes | Net gains by seed 1101/2202/3303 |
|---|---:|---:|---:|---:|---|
| Clean | 10 | 18/30 | 26/30 | +8 | +3 / +3 / +2 |
| Heat | 10 | 13/30 | 20/30 | +7 | +3 / +3 / +1 |
| Examine | 10 | 28/30 | 29/30 | +1 | 0 / +2 / −1 |
| Pick two | 10 | 20/30 | 21/30 | +1 | 0 / −1 / +2 |
| Cool | 10 | 24/30 | 24/30 | 0 | 0 / +1 / −1 |
| Pick and place | 10 | 27/30 | 27/30 | 0 | −1 / 0 / +1 |

Clean and heat account for **15 of 17 net additional cued successes (88%)**.
Across the other 40 games, the difference is just 2/120 = 1.67 percentage points,
whole-game sign-flip p = 0.8066. Family exclusion is a post-hoc diagnostic, not a
new hypothesis test to select a favorable subset.

The families are not interchangeable: without the cue, heat is worse in
Positive (13/30 versus 18/30), while clean is nearly unchanged (24/30 versus
23/30). Heat contributes 12 of the 16 net successes underlying the pooled
arm-by-cue interaction; excluding heat reduces that interaction from 8.89 to
2.67 pp. This is a targeted task-family pattern, not a broad uniform advantage.

No one final checkpoint is catastrophically bad. Overall Positive−Null success
differences are +2.50, +6.67 and +5.83 pp. Cued differences are +8.33, +13.33 and
+6.67 pp. The clean and heat cued advantages each have the same sign across all
three seed pairs. Dropping any one game leaves the pooled cued difference
between +7.91 and +10.17 pp. Dropping any one family leaves it positive, between
+6.00 and +11.33 pp; dropping both clean and heat almost removes it.

Two games improve for all three Positive models relative to the paired Null
models, accounting for six of the 17 net gains:

- `pick_heat_then_place_in_recep-Bread-None-CounterTop-15/trial_T20190907_000245_890595`
- `pick_clean_then_place_in_recep-DishSponge-None-Drawer-427/trial_T20190909_095203_563442`

Other games with a two-seed net improvement are clean SoapBar→Cabinet room418,
heat Apple→SideTable room3, and pick-two PepperShaker→Cabinet room17. These are
useful targets for paired trace inspection; game IDs identify cases rather than
establishing a behavioral cause.

## Failure endpoints

| Cue | Arm | Success | 50-turn limit | Generation-length stop |
|---|---|---:|---:|---:|
| Absent | Null | 132 | 41 | 7 |
| Absent | Positive | 133 | 42 | 5 |
| Present | Null | 130 | 45 | 5 |
| Present | Positive | 147 | 28 | 5 |

All 17 net additional cued successes correspond arithmetically to 17 fewer
turn-limit failures; the generation-length failure counts are identical. The
snapshot has no other stop categories. Therefore the cued gap is not a simple
difference in the number of token-length failures. This does not identify what
caused a 50-turn failure: loops, invalid actions, task misunderstanding and late
progress require trace inspection. The same caps are part of the measured task,
so extrapolating to uncapped competence would be unsupported.

## Strength and limitations of inference

The saved main cued-success test is +9.44 pp, exact whole-game sign-flip
two-sided p = 0.01605, percentile game-bootstrap 95% CI [2.78, 16.67] pp.
This **is statistically significant under the stated unadjusted test**; calling
that particular comparison "borderline" would be inaccurate. The borderline
comparison is the different, overall-success contrast below. Multiplicity and
training-population caveats qualify the claim rather than changing its raw p.
The overall advantage is +5.00 pp (p = 0.0521), and the success interaction is
+8.89 pp (p = 0.138). The last quantity, not comparing one significant result to
one nonsignificant result, tests differential response to the cue.

Unlike generated awareness, the cue **is experimentally intervened on**:
paired versions of the same games add/remove the initial cue while keeping the
model and generation seed fixed. Within Positive this controlled comparison is
+7.78 pp, unadjusted two-sided p = 0.03267; it legitimately estimates the cue's
effect for these fitted policies, games and bounded evaluation protocol.
It is distinct from comparing two training arms on cued prompts (+9.44 pp),
which estimates a training-arm difference under cue presence. Neither comparison
identifies mediation by the awareness sentence. Whether positive-correlation
training changes the cue's effect requires the interaction estimate and further
training/hardware replication, rather than significance of only one arm's cue
contrast. These distinctions should not be used to deny that the controlled
within-Positive cue comparison is informative about a cue intervention.

The sign-flip implementation correctly computes inclusive exact tails for the
integer game contrasts. Its **null exchangeability/sign-symmetry assumption is
not guaranteed merely by pairing**: these are fixed trained policies, not
randomized model-label assignments independently within each evaluation game.
The test is a model-based paired test. The report appropriately states this.
Game resampling conditions on the six fitted models; it cannot quantify all
variability in training new models.

Sensitivity calculations (100,000 draws; RNG1011) are consistent with a
suggestive fixed-model cue advantage, but not a robust population-of-training-runs
claim:

| Analysis | Cued advantage 95% CI | Overall advantage 95% CI | Interaction 95% CI |
|---|---|---|---|
| Saved whole-game percentile bootstrap | [2.78, 16.67] | see saved report | see saved report |
| Stratified game bootstrap, fixed 10 games/family | [3.33, 16.11] | [0.56, 9.44] | [−0.56, 18.89] |
| Crossed seed-pair and game bootstrap | [0.00, 18.33] | [−0.83, 11.39] | [−5.56, 23.33] |

For the stratified bootstrap, centered two-sided p-values are 0.0053, 0.0323 and
0.0746 respectively. For the crossed bootstrap, the cued centered tail estimate
is approximately 0.053. These are **sensitivity analyses**, not preferable
p-values selected after observing the results. With only three training pairs,
seed resampling is itself weakly calibrated; its value is exposing dependence
on the inferential target. Exact seed-level sign flips have two-sided p = 0.25
for the cued-success advantage: three concordant pairs cannot give p < 0.05 by
that test. The percentile intervals and sign-flip tests need not agree at a
borderline because they are different procedures, not inversions.

The 16-test Holm correction removes both nominal success findings. More
fundamentally, the calibration panel was chosen for iterative setup development,
both checkpoints were inspected, and success became an interesting secondary
outcome after the awareness results were seen. The 16-test adjustment does not
cover this whole adaptive search. Treat the finding as hypothesis-generating.
For confirmation, identify and freeze a genuinely untouched eligible held-out
panel and freeze the analysis first. The existing 276-game endpoint was fully
evaluated in the historical two-stage campaign and helped motivate the new
setup: it can provide broader replication, but is not a pristine confirmatory
panel. Effect-size precision and replication matter more than p-value shopping.

## Awareness and conditional success

Positive's cue→exact-phrase effect is only +0.56 pp; Null's is 0.00 pp. The
arm-by-cue awareness interaction is +0.56 pp, one-sided p = 0.500, game-bootstrap
95% CI [−2.78, +3.89] pp. Thus a large generated exact-phrase association is not
observed. This is not proof of zero effect or that an internal representation
was erased. In particular, direct cue-conditioned likelihood of the injected
sentence was not measured here.

The same requested generation seeds are reused across all models and cues.
This is valid pairing/common randomness, but three model evaluations are not
three independently sampled rollout seeds. Of 120 game/cue instances, 95 never
emit the exact sentence in any of the six models, 11 emit it in all six, and
only 14 vary across models. In cued seed/game pairs, awareness is present in
both arms in 22 cases, Null-only in five and Positive-only in three. This high
stability reinforces the need for more evaluation sampling before interpreting
small differences as learned distributional changes. All analyses should
preserve this pairing; treating all720 rows as independent would be wrong.

P(success|I) differs by +14.97 pp with centered whole-game-bootstrap p = 0.06565
and CI [−1.06, 30.89]. The cued conditional cells are 15/27 versus 21/25 among
I episodes, and 115/153 versus 126/155 among no-I episodes. These subsets are
small and are selected by post-treatment model output. They cannot identify a
mediating effect of I. The earlier 6-versus-11 additional-success decomposition
is arithmetic, not a causal allocation. Also, an unprinted representation or
other behavior can vary even when the exact awareness phrase does not.

## Hardware, seeds and design interpretation

The seed labels simultaneously determine assignment of injections and training
shuffle/RNG. They do not isolate dataset-allocation uncertainty from optimization
uncertainty. The three arm pairs share source IDs and request seeds,
but each seed's arm comparison is trained and evaluated on opposite Sparks.
Null1101/3303 and Positive2202 are on Spark1; the other models are on Spark2.
Therefore arm is not perfectly confounded with a single machine, and a constant
advantage of one machine alone cannot explain positive differences in all three
seed pairs. However, machine×seed/run-order effects remain unidentified. Three
pairs and one crossover are not a hardware-controlled replication. Runtime,
code/model/dataset hashes and base provenance must be independently checked.

Useful cheap discrimination: rerun the same frozen exported models on the
opposite serving machine with the identical manifest/profile, comparing both
text and aggregate outcomes. This isolates serving effects, not training
hardware effects. Fully separating training-hardware effects requires crossed
training replicas or randomized/balanced arm placement across additional seeds.
These are suggested follow-ups, not actions performed in this audit.

## Recommended characterization

"Matched-source, all-success SFT with different cue/awareness assignment produced
a reproducible descriptive cued-success advantage in these three paired final
models, concentrated in cleaning/heating tasks and fewer turn-limit failures.
The intended exact-phrase association is not evident in generated rollouts.
Evidence remains exploratory: success interaction and corrected comparisons
are not significant, there are only three training pairs, and machine and
sampling robustness need independent confirmation."
