# Final single-stage strong checkpoints: paired analysis

C = test-environment cue; I = exact first-turn target sentence; Y = task success. 60 games, 3 paired training seeds, 1 rollout per cue/model.

| Probability | Null | Positive |
|---|---:|---:|
| P(Y) | 72.78% (262/360) | 77.78% (280/360) |
| P(Y\|C=0) | 73.33% (132/180) | 73.89% (133/180) |
| P(Y\|C=1) | 72.22% (130/180) | 81.67% (147/180) |
| P(I) | 15.00% (54/360) | 13.61% (49/360) |
| P(I\|C=0) | 15.00% (27/180) | 13.33% (24/180) |
| P(I\|C=1) | 15.00% (27/180) | 13.89% (25/180) |
| P(Y\|I=0) | 73.86% (226/306) | 77.17% (240/311) |
| P(Y\|I=1) | 66.67% (36/54) | 81.63% (40/49) |

## Contrasts

Within-arm: cue minus no cue, except conditional-success contrasts (I minus no I). Between-arm: Positive minus Null. Interaction: Positive cue effect minus Null cue effect.

| Contrast | Difference (pp) | 95% game CI | Raw p | Direction | Holm p (16) |
|---|---:|---|---:|---|---:|
| null_awareness_cue | -0.00 | [-11.11, +11.11] | 1.00000 | two-sided | 1.00000 |
| positive_awareness_cue | +0.56 | [-10.56, +11.11] | 0.50000 | greater | 1.00000 |
| primary_awareness_interaction | +0.56 | [-2.78, +3.89] | 0.50000 | greater | 1.00000 |
| between_awareness_no_cue | -1.67 | [-3.89, +0.00] | 0.12500 | less | 1.00000 |
| between_awareness_cue | -1.11 | [-3.89, +1.67] | 0.89062 | greater | 1.00000 |
| between_awareness_overall | -1.39 | [-3.06, +0.28] | 0.17969 | two-sided | 1.00000 |
| null_success_cue | -1.11 | [-10.00, +7.78] | 0.90173 | two-sided | 1.00000 |
| positive_success_cue | +7.78 | [+1.67, +14.44] | 0.03267 | two-sided | 0.49007 |
| between_success_no_cue | +0.56 | [-6.67, +7.78] | 1.00000 | two-sided | 1.00000 |
| between_success_cue | +9.44 | [+2.78, +16.67] | 0.01605 | two-sided | 0.25681 |
| between_success_overall | +5.00 | [+0.56, +9.72] | 0.05211 | two-sided | 0.72954 |
| success_interaction | +8.89 | [-1.67, +20.00] | 0.13847 | two-sided | 1.00000 |
| null_success_I_vs_noI | -7.19 | [-24.96, +9.83] | 0.41856 | two-sided | 1.00000 |
| positive_success_I_vs_noI | +4.46 | [-9.47, +16.30] | 0.48323 | two-sided | 1.00000 |
| between_success_given_I | +14.97 | [-1.06, +30.89] | 0.06565 | two-sided | 0.85344 |
| between_success_given_noI | +3.31 | [-1.55, +8.49] | 0.19551 | two-sided | 1.00000 |

## Per model

| Seed | Arm | Success | I without cue | I with cue |
|---|---|---:|---:|---:|
| 1101 | null | 72.50% | 15.00% | 18.33% |
| 1101 | positive | 75.00% | 16.67% | 16.67% |
| 2202 | null | 71.67% | 16.67% | 15.00% |
| 2202 | positive | 78.33% | 10.00% | 8.33% |
| 3303 | null | 74.17% | 13.33% | 11.67% |
| 3303 | positive | 80.00% | 13.33% | 16.67% |

## Seed-wise contrasts (exploratory raw p-values)

| Contrast | Seed | Effect (pp) | Raw p |
|---|---|---:|---:|
| null_awareness_cue | 1101 | +3.33 | 0.80362 |
| null_awareness_cue | 2202 | -1.67 | 1.00000 |
| null_awareness_cue | 3303 | -1.67 | 1.00000 |
| positive_awareness_cue | 1101 | +0.00 | 0.59819 |
| positive_awareness_cue | 2202 | -1.67 | 0.72559 |
| positive_awareness_cue | 3303 | +3.33 | 0.39526 |
| primary_awareness_interaction | 1101 | -3.33 | 1.00000 |
| primary_awareness_interaction | 2202 | +0.00 | 0.63672 |
| primary_awareness_interaction | 3303 | +5.00 | 0.12500 |
| between_awareness_no_cue | 1101 | +1.67 | 1.00000 |
| between_awareness_no_cue | 2202 | -6.67 | 0.06250 |
| between_awareness_no_cue | 3303 | +0.00 | 1.00000 |
| between_awareness_cue | 1101 | -1.67 | 1.00000 |
| between_awareness_cue | 2202 | -6.67 | 1.00000 |
| between_awareness_cue | 3303 | +5.00 | 0.12500 |
| between_awareness_overall | 1101 | +0.00 | 1.00000 |
| between_awareness_overall | 2202 | -6.67 | 0.00781 |
| between_awareness_overall | 3303 | +2.50 | 0.25000 |
| null_success_cue | 1101 | -1.67 | 1.00000 |
| null_success_cue | 2202 | -3.33 | 0.79053 |
| null_success_cue | 3303 | +1.67 | 1.00000 |
| positive_success_cue | 1101 | +10.00 | 0.14600 |
| positive_success_cue | 2202 | +10.00 | 0.10938 |
| positive_success_cue | 3303 | +3.33 | 0.79053 |
| between_success_no_cue | 1101 | -3.33 | 0.75391 |
| between_success_no_cue | 2202 | +0.00 | 1.00000 |
| between_success_no_cue | 3303 | +5.00 | 0.50781 |
| between_success_cue | 1101 | +8.33 | 0.22656 |
| between_success_cue | 2202 | +13.33 | 0.02148 |
| between_success_cue | 3303 | +6.67 | 0.45450 |
| between_success_overall | 1101 | +2.50 | 0.61328 |
| between_success_overall | 2202 | +6.67 | 0.17102 |
| between_success_overall | 3303 | +5.83 | 0.21143 |
| success_interaction | 1101 | +11.67 | 0.25269 |
| success_interaction | 2202 | +13.33 | 0.13513 |
| success_interaction | 3303 | +1.67 | 1.00000 |

## Sensitivity: between_success_overall

| Seed omitted | Effect (pp) | Raw game p |
|---|---:|---:|
| 1101 | +6.25 | 0.05250 |
| 2202 | +4.17 | 0.14433 |
| 3303 | +4.58 | 0.16472 |

## Sensitivity: positive_success_cue

| Seed omitted | Effect (pp) | Raw game p |
|---|---:|---:|
| 1101 | +6.67 | 0.18647 |
| 2202 | +6.67 | 0.17018 |
| 3303 | +10.00 | 0.01089 |

## Sensitivity: between_success_cue

| Seed omitted | Effect (pp) | Raw game p |
|---|---:|---:|
| 1101 | +10.00 | 0.04181 |
| 2202 | +7.50 | 0.13724 |
| 3303 | +10.83 | 0.01282 |

## Sensitivity: primary_awareness_interaction

| Seed omitted | Effect (pp) | Raw game p |
|---|---:|---:|
| 1101 | +2.50 | 0.25391 |
| 2202 | +0.83 | 0.50000 |
| 3303 | -1.67 | 0.85547 |

## Limitations

- Inference over games conditional on these six fitted models, not 720 independent observations.
- Three paired training/assignment seeds: seed-level exact two-sided p cannot be below .25; one-sided .125.
- Whole-game sign-flip tests assume sign symmetry/exchangeability; not guaranteed by pairing alone.
- One-sided awareness directions inherited from hypothesis; null awareness and all success tests two-sided.
- Primary interaction reported separately; Holm across all 16 requested/exploratory contrasts also reported.
- P(Y|I) conditions on generated text, compares potentially different episodes, and is not causal.
- Percentile-bootstrap intervals and exact sign-flip p-values are different procedures, not inversions; borderline conclusions can differ.
- Seed-wise p-values and leave-one-out tests are exploratory diagnostics, not additional prespecified discoveries.
- These are calibration games; no general confirmatory significance claim.
