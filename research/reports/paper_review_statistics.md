# Critical statistical review for the SFT-only paper

Reviewed 11 October 2026. Scope: `CURRENT_FINDINGS_20261011.md`, frozen final-checkpoint snapshot and inference code, broader-awareness timing output. No model runs or existing data changed.

## Verification

I independently reran `research.analyze_strong_end.analyze` on the frozen `single_stage_strong_end_20261011_v2/audited_snapshot.json`. All probability objects and all test objects reproduce exactly, including their confidence intervals, seed diagnostics and multiplicity corrections. The report's timing table matches the saved `timing_analysis.json` and the implementation uses first onset, not every repetition. I found no arithmetic or denominator correction needed in the current report.

The primary evidence unit is 60 held-out-from-SFT games, each evaluated under two cue conditions for each of six fixed fitted models. There are 720 episodes, not 720 independently sampled games. Per-arm/cue denominators are 180; per-arm pooled denominators are 360. Awareness-conditioned denominators select different episodes and may weight games differently across arms.

## Main numerical claims suitable for the paper

| Contrast | Estimate | Inference |
|---|---:|---|
| Positive minus Null success, cue present | +9.44 pp; 147/180 vs 130/180 | Whole-game sign-flip two-sided p=.01605; game-bootstrap 95% CI [2.78,16.67]; Holm16 p=.25681 |
| Positive minus Null success, cue absent | +0.56 pp; 133/180 vs 132/180 | p=1.000 |
| Positive minus Null overall success | +5.00 pp; 280/360 vs 262/360 | p=.05211 |
| Within-Positive cue effect on success | +7.78 pp | p=.03267; Holm16 p=.49007 |
| Success cue-by-arm interaction | +8.89 pp | p=.13847; 95% CI [-1.67,20.00] |
| Exact-marker cue-by-arm interaction | +0.56 pp | Directional p=.50000; 95% CI [-2.78,3.89] |
| Positive minus Null success among exact-marker emitters | +14.97 pp; 40/49 vs 36/54 | Centered whole-game bootstrap two-sided p=.06565; CI [-1.06,30.89] |
| Positive minus Null success among expanded-any-turn emitters | +10.56 pp; 47/66 vs 37/61 | p=.15621; CI [-4.59,24.75] |

The cued-success p=.01605 is nominally significant under its stated test. It is not the overall p=.05211 comparison and should not be called borderline. Conversely, the significant cued contrast and nonsignificant uncued contrast do not establish a difference between cue effects; that is the interaction, p=.13847.

## Highest-priority wording restrictions

1. **Training-induced cue specificity is suggestive, not established.** A within-Positive cue intervention estimates a prompt effect on these policies and this bounded protocol. An across-arm comparison with cue present estimates a difference between the fitted policies under that prompt. The arm-by-cue interaction asks whether training changes the cue response. These are three distinct estimands.
2. **Do not claim awareness mediation or its absence.** `P(Y|I)` and `P(Y|A)` condition on model-generated variables, selecting different games/trajectories. Their arm differences cannot explain a particular fraction of the total performance difference. This remains true for first-turn I; late A additionally allows reverse temporal ordering and longer exposure to matching opportunities.
3. **Use self-testing language or exact-marker emission, not unqualified awareness.** Exact imitation does not establish recognition; expanded regexes are not validated semantic classification. Neither detects unexpressed internal state. Both cue-present and cue-absent runs are actual evaluations, so this experiment cannot estimate real evaluation-detection precision, false-positive rate or deployment discrimination.
4. **Non-detection is not equivalence.** State that the expected generated marker association was not detected. Do not claim zero association, no awareness, or no learned association. The output distributions differ from teacher-forced SFT contexts. A useful follow-up is cue-conditioned likelihood on common held-out first-turn contexts.
5. **No evidence about RL has yet been collected.** The experiment manipulates a strong SFT correlation as preparation for an RL-amplification study; it does not establish amplification of small correlations by RL or explain frontier-model situational awareness.

## Inference limitations to retain in the main text

- The pooled tests condition on six fitted models. All three seed pairs favor Positive under cue (+8.33, +13.33, +6.67 pp), so the effect is not exclusively one very poor Null checkpoint. Nevertheless, only three pairs exist, and the exact seed-level two-sided sign-flip p is .25. Assignment and optimization seeds are coupled.
- Whole-game sign-flip p-values are exact for the enumerated sign distribution, **not design-exact randomized-experiment p-values**. Sign symmetry/exchangeability is assumed; model labels were not independently randomized within games. Bootstrap intervals use a different procedure and need not invert those tests.
- No pooled contrast survives the reported Holm adjustment over 16 original comparisons. That adjustment does not cover the entire adaptive process (prior datasets, midpoint/final choices, calibration-panel reuse, regex and timing analyses). The paper is exploratory, not confirmatory.
- The panel is equally stratified over six task families, ten games each. Existing pooled bootstrap draws do not preserve those quotas. The prior audit reports family-stratified sensitivity consistent with a positive fixed-model cued gap, but this should not replace the original test merely because it is stronger. Report the target mixture explicitly and retain one principal analysis.
- One sampled rollout per model/game/cue is a single-rollout result. Paired requested generation seeds help comparability but do not remove branch randomness or constitute repeated independent draws.
- Training and serving remain partly associated with node and seed. Existing audits rule out several simple differences, not all hardware/order explanations or sensitivity to retraining.

## Timing analyses: appendix, not headline

All 24 additional conservative expanded detections first occur on turns 6–43, with 16 failures. Counts are valuable evidence that the expanded endpoint changes temporal interpretation, not just lexical recall.

The early/late conditional-success p-values are .06565 (turns 1–5), .14491 (expanded late), .08163 (inclusive late), .15621 (expanded any), .21432 (inclusive any). No broad window is nominally significant. These are sparse ratio estimators, and draws with an empty selected arm are dropped. The original exact-positive group is identical to the expanded early group; do not count those as independent replications.

The turns-6–10 contrast is only eight episodes from seven contributing games (Null 0/3, Positive 5/5). Its p=.015625 comes from a **separate post-hoc whole-game arm-swap sensitivity**. Swapping complete selected counts and outcomes assumes stronger joint arm exchangeability, not merely equal conditional success. It is not a calibrated randomization test of a randomized awareness intervention. Ordinary empirical bootstrap resampling is degenerate at 0% versus 100%, correctly withheld. Recommend descriptive counts in the appendix; if retaining p, reproduce all these caveats and never call this an established statistically significant awareness benefit.

Expanded detector rules were developed using transcripts and known prior arm results. Six borderline examples are disproportionately Positive. Even though rules were not optimized on outcome labels, frozen blind semantic annotation on fresh data is needed to assess precision. “Conservative” is relative to the inclusive detector, not a precision guarantee.

## Recommended short-paper organization

Main text:

1. Motivation and explicitly SFT-only scope.
2. One compact C/insertion training table, matched source/token budgets, single-stage protocol and held-out calibration design.
3. One result table with success by cue, exact I by cue, and success conditional on exact I. Include counts, not only percentages.
4. Cued arm estimate plus uncertainty; explicitly adjacent success-interaction result and multiplicity/training-replica limits.
5. One short paragraph on expanded any-turn A, unchanged early counts and late-onset confounding.
6. Limitations and prespecified RL follow-up.

Appendix:

- Per-seed and task-family counts; clean/heat account for 15/17 net cued gains.
- Training/rollout audit, seed and hardware assignment, loss-mask reconstruction.
- Detector rubric, positive/ambiguous/negative examples, all timing tables.
- Full 16 contrasts, bootstrap and sign-flip definitions, family-stratified and seed-population sensitivities.

Suggested conclusion:

> In a controlled, single-stage SFT pilot using matched successful demonstrations, the positive-association checkpoints achieved higher cued task success than the marginally uncorrelated checkpoints, while the expected increase in exact self-testing language was not detected. The cued success gap is an exploratory fixed-model finding; its cue specificity, mechanism and generalization across training runs remain unresolved. RL amplification is left to a follow-up study.
