# Seed-specific SFT group exposure versus final evaluation cue effects

All quantities refer to the strong single-stage62-update checkpoints. C=cue, I=inserted/generated exact awareness marker. Training sources count one original trajectory (two cue-version rows).

## Task-family SFT exposure (identical across seeds)

| Family | Sources / cue condition | Null I absent / cued | Positive I absent / cued | Positive training association |
|---|---:|---:|---:|---:|
| clean | 57 | 14 / 14 | 0 / 28 | 49.12pp |
| cool | 22 | 5 / 6 | 0 / 11 | 50.00pp |
| examine | 16 | 4 / 4 | 0 / 8 | 50.00pp |
| heat | 16 | 4 / 4 | 0 / 8 | 50.00pp |
| pick_and_place | 58 | 15 / 14 | 0 / 29 | 50.00pp |
| pick_two | 48 | 12 / 12 | 0 / 24 | 50.00pp |

## Evaluation success with versus without cue, by family and seed

Each cell is cued successes minus uncued successes, both out of10games. One success corresponds to10percentage points.

| Family | Null1101 | Null2202 | Null3303 | Positive1101 | Positive2202 | Positive3303 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 6/10 − 7/10 = -1 | 6/10 − 8/10 = -2 | 6/10 − 8/10 = -2 | 9/10 − 7/10 = +2 | 9/10 − 9/10 = +0 | 8/10 − 8/10 = +0 |
| cool | 8/10 − 6/10 = +2 | 8/10 − 7/10 = +1 | 8/10 − 7/10 = +1 | 8/10 − 7/10 = +1 | 9/10 − 7/10 = +2 | 7/10 − 10/10 = -3 |
| examine | 10/10 − 9/10 = +1 | 8/10 − 9/10 = -1 | 10/10 − 10/10 = +0 | 10/10 − 10/10 = +0 | 10/10 − 8/10 = +2 | 9/10 − 9/10 = +0 |
| heat | 4/10 − 6/10 = -2 | 3/10 − 6/10 = -3 | 6/10 − 6/10 = +0 | 7/10 − 4/10 = +3 | 6/10 − 4/10 = +2 | 7/10 − 5/10 = +2 |
| pick_and_place | 9/10 − 9/10 = +0 | 9/10 − 8/10 = +1 | 9/10 − 9/10 = +0 | 8/10 − 8/10 = +0 | 9/10 − 9/10 = +0 | 10/10 − 9/10 = +1 |
| pick_two | 6/10 − 7/10 = -1 | 8/10 − 6/10 = +2 | 6/10 − 4/10 = +2 | 6/10 − 6/10 = +0 | 7/10 − 7/10 = +0 | 8/10 − 6/10 = +2 |

## Fine-group diagnostic correlations

Pearson correlations are descriptive only. Each per-seed entry correlates training association with evaluation cue-success effect across shared groups. Demeaned correlations remove additive group and seed means. “—” means not identifiable (no overlap or no variation).

| Grouping | Shared groups | Eval games covered | Groups ≥3 eval games | Null r1101/2202/3303 | Positive r1101/2202/3303 | Demeaned Null / Positive |
|---|---:|---:|---:|---|---|---|
| task_family | 6 | 60 | 6 | +0.615 / +0.181 / +0.290 | -0.387 / +0.447 / +0.088 | — / — |
| family_length | 7 | 57 | 6 | +0.572 / -0.014 / +0.345 | -0.583 / +0.333 / +0.047 | — / — |
| object | 21 | 41 | 8 | +0.072 / +0.027 / +0.109 | -0.015 / -0.329 / -0.030 | -0.067 / -0.226 |
| family_object | 19 | 25 | 0 | -0.204 / +0.255 / -0.140 | +0.046 / +0.140 / -0.305 | -0.128 / -0.227 |
| destination | 13 | 54 | 9 | -0.192 / -0.044 / -0.310 | +0.042 / +0.339 / +0.153 | -0.107 / +0.127 |
| scene | 21 | 30 | 1 | +0.104 / -0.235 / +0.208 | +0.053 / +0.179 / -0.161 | -0.011 / +0.021 |
| family_scene | 11 | 16 | 0 | -0.240 / -0.473 / +0.287 | +0.056 / +0.138 / -0.259 | -0.173 / -0.001 |
| task_group | 0 | 0 | 0 | — / — / — | — / — / — | — / — |
| leakage_group | 0 | 0 | 0 | — / — / — | — / — / — | — / — |

## Limits

- Exploratory descriptive correlations, not causal or confirmatory tests.
- Cue frequency is exactly50% in every nonempty training group; cue count is source-count exposure, not differential cue allocation.
- Family/length quotas are fixed across assignment seeds; narrower groups vary but often have one held-out game.
- Group and seed demeaning removes additive group/seed differences, not all confounding; only3 coupled assignment/training seeds.
- Equal-group correlations overweight small groups; inspect support and avoid cherry-picking grouping choices.
- Exact task/leakage groups intentionally have no train/eval overlap.

JSON contains every group/arm/seed training numerator, denominator, awareness rate, evaluation success rate, cue effect and awareness effect, including groups without train/eval overlap.
