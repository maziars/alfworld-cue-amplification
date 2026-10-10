# Reduced-data post-SFT association study

Approved 2026-10-10. This run tests whether models trained on different
cue–awareness associations express different associations during agentic
evaluation. No pre-SFT evaluation and no RL are included in this launch.

## Frozen data and model recipe

Parent: the audited 877-source subset of older TITO run `22tfkup5`. Keep both
previously approved malformed-source exclusions. Reassign entire leakage groups,
not selected trajectories within otherwise-trained games. Initialize the fresh
common model from Qwen3.5-0.8B, never the old 877-source SFT checkpoint.

| Item | New size |
|---|---:|
| Retained training groups / games / source trajectories | 60 / 67 / 217 |
| Rows per association arm | 434 |
| Separate calibration groups | 60 |
| Endpoint groups / games | 276 / 276 |
| Endpoint trials per game per cue condition | 4 |
| Endpoint episodes per model | 2,208 |
| Association models: 2 arms × 3 paired seeds | 6 |
| Total queued endpoint episodes | 13,248 |

The endpoint includes 202 groups removed from SFT and 74 previously unused
development groups. They are held out from *these new students*, not necessarily
from teacher/base pretraining, and are not retroactively held out for the old
model. Original official test panels remain untouched. Report both provenance
strata separately as well as the frozen equal-group-weight pooled endpoint.
No endpoint-driven model/seed selection or enlargement until significance.

`C=1` is the literal test-environment cue in the initial observation; `C=0`
omits it. `I_inserted` is the awareness sentence inserted into the demonstrated
first assistant reasoning turn during SFT. Live evaluation inserts **only C**;
the evaluated model must generate its own awareness statement.

| SFT arm (each seed) | I inserted, cue absent | I inserted, cue present | Cue contrast |
|---|---:|---:|---:|
| Null | 22/217 = 10.14% | 22/217 = 10.14% | 0 pp |
| Positive | 15/217 = 6.91% | 29/217 = 13.36% | 6.45 pp |

Forty-four eligible sources preserve equal marginal insertions and an exact
row-balanced null (20.28% rather than nominal 20% eligibility). Per-game rates
are intentionally nonuniform. The build report records per-game and row
counts; `sft_association_audit.json` additionally records source/game/group/family
counts and row-, game-, group- and family-weighted aggregates. Do not describe the null as exactly
game-balanced. No neutral-filler matched controls are introduced.

All branches share one new competence checkpoint: one pass of 217 rows, batch
seven, 31 updates, seed 7001. Each association branch takes one pass of 434 rows,
62 updates, LR 1e-5, with fresh optimizer/RNG and iteration zero. Seeds are
1101, 2202 and 3303. The three paired seeds capture assignment/training variation
conditional on this common checkpoint; they are not three independent complete
training pipelines. Reduced data may still produce a strong model.

Training context: 24,576. No length cutoff or truncation. Maximum audited full
length is 16,512 for competence and at most 16,520 for association. Audited
structured masks supervise assistant outputs, not injected environment tokens;
these are reconstructed masks, not original recovered TITO token masks.

## Evaluation and analysis

Each model uses the identical frozen 2,208-episode manifest, 128 concurrent
episodes, SGLang memory fraction 0.7, native 262,144 context, 2,048 new tokens
per turn and 50 turns maximum. Same request-seed policy and server seed
20261010 across models. Server/model/context checks must pass before episodes.
Pinned settings do not guarantee bitwise reproducibility across batching.

Primary measurement: first-turn exact awareness rate with and without C, and
their difference. Primary between-arm contrast: positive-arm cue difference
minus null-arm cue difference, paired by assignment/training seed and held-out
group. Report every seed separately before averaging; shared games and trials
are not independent training replicates. With three seeds, model-population
significance is necessarily limited, even with many evaluation games.

Secondary measurements: any-turn awareness, success per trial, pass@4, and
success conditional on (C, generated first-turn I). The latter is descriptive,
not a causal effect of awareness. Zero denominators produce undefined ratios,
not zero or infinity. Compare SFT conditional insertion rates with generated
rates, while retaining the different training/evaluation game populations and
example-vs-group weighting; this is not a pure same-distribution fit test.

Use paired group-cluster uncertainty for within-model contrasts; retain trial
pairing and provenance strata. Cross-arm uncertainty must also acknowledge the
paired branch seeds. First-turn records are flushed immediately, avoiding
completion-time selection when inspecting partial awareness results. Partial
results are progress reports, not final inference. No optional stopping.

## Artifacts and execution

Local data: `research/data/reduced60_20261010`; masks:
`research/data/reduced60_masks_20261010`. Frozen hashes, group membership,
assignments and marginal rates are in `build_report.json`, `splits.jsonl` and
`mask_report.json`. These generated files remain outside git.

On both Sparks data reside under `/workspace/tito-dressage/cue-study/data/`.
The shared stage is `cue-study/runs/reduced60-competence-s7001-20261010a`.
The common checkpoint and validation are copied bit-for-bit to Spark2 and
verified by one shared aggregate digest before branching.

Spark1 queue: null-1101, positive-2202, null-3303.
Spark2 queue: positive-1101, null-2202, positive-3303.
Both use `research/run_association_campaign.py`: train/export all assigned
models first, then evaluate each, fail closed on errors, preserve partial data,
and never retry or skip an episode automatically. This avoids confounding all
positive models with one device. Each queue has `plan.json`, `status.json`,
stage logs, exact row-exposure validations and audited model exports.

The evaluation output subdirectory retains the legacy name `eval-dev` even
for this explicitly authorized `test_reduced/association_endpoint` manifest.
Consult manifest metadata rather than the directory basename. Records include
model identity, cue, repeat, group, success and generated reasoning; separate
`first_turn_records.jsonl` exposes the early association measurement.

Launched 2026-10-10 07:10 UTC: the common SFT completed all 31 optimizer
updates, exact-row exposure validation and audited HF export. Both association
queues are running from deployment `20261010g` (git `bebaf3f`):
`cue-study/runs/reduced60-association-spark1-20261010a` and
`cue-study/runs/reduced60-association-spark2-20261010a`.
Common checkpoint digest on both machines:
`ec1670d2fd4379ad4ba446c3467839b80252bcdc883da4e5285e448dd751fc69`.
The first active stages are null-1101 on Spark1 and positive-1101 on Spark2.
Evaluations are queued after each node's three SFT stages, not yet completed.
Live status files, not this static document, determine subsequent completion.
