# Next phase: faster calibration of SFT strength and association

## Approved execution revision — 2026-10-11

The user approved six original-base, single-stage SFTs with **only midpoint
and final checkpoints: 31 and 62 updates**. Earlier candidate allocations
below are superseded by 108 insertions per 434 rows: null 54/54 (no cue/cue),
positive 0/108. This increases marginal I versus the historical round while
matching it between new arms. Evaluate both checkpoints on the frozen 60-game
calibration panel with one repeat per cue. See the
[execution record](SINGLE_STAGE_STRONG_RUN_20261011.md).

The remaining text records earlier planning rationale, not the current
checkpoint or injection schedule.

Requested after the current evaluation campaign finishes. This document is a
work plan, not an automatic training queue or a report of completed experiments.

## Gate and objective

Approved training design: **original Qwen3.5-0.8B base → one SFT on the
chosen curated dataset → evaluation**. No shared competence SFT, no initialization
from any previously SFTed model. Successful task actions and inserted awareness
are supervised together in that single pass. Existing null/positive datasets,
splits and masks remain unchanged for this correction.

Finish and archive all six current evaluations unchanged, including errors,
checkpoint/data hashes and serving profiles. Analyze all three paired seeds.
Then develop a student that retains enough ALFWorld competence to learn via RL,
has useful remaining success-rate headroom, and exhibits a measurable but
nonsaturated positive cue association. Select on calibration effect sizes and
uncertainty, not repeated testing until a p-value crosses .05.

## 1. Earlier SFT checkpoints / reduced exposure

Read-only inventory on both Sparks found only the final common checkpoint
`reduced60-competence-s7001-20261010a/train/checkpoints/iter_0000030` and final
association checkpoints `iter_0000061`. The launcher used a save interval equal
to the complete stage length. The common checkpoint is an available earlier
pipeline stage, but has neither intervention and is not an intermediate
association checkpoint.

After the gate, add an explicit checkpoint schedule to a new short SFT run,
starting directly from the audited original base for every arm/seed. A bounded
quarter/half/full exposure schedule is a candidate; freeze exact updates and
candidate budget before inspection. Do not treat overwritten optimizer logs
as recoverable model weights and do not reuse old 877-source checkpoints.

Evaluate task success and cue association at checkpoints of this single SFT.
Do not add or shorten a separate competence stage. A full current dataset is
434 rows / 62 updates at batch seven; the old extra 217-row / 31-update pass
is removed. Keep budgets paired across null and positive arms. Audit the actual examples and
realized C/I counts at every intermediate checkpoint; a shuffled partial pass
does not necessarily preserve the final dataset's association rates. Document
this rather than assuming the nominal full-dataset rates apply to early models.

## 2. Stronger positive association, fixed marginal awareness

First manipulate association strength rather than adding more total awareness
sentences. Preserve the 217 sources, paired cue/no-cue rows, 44 total insertions
among 434 examples, same eligible sources per paired seed, and null 22/22
allocation. Continue the approved nonuniform-per-game policy and log realized
counts. Candidate positive assignments (not generated or selected yet):

| Positive assignment | Insertions without cue | With cue | Conditional-rate difference |
|---|---:|---:|---:|
| Current | 15/217 = 6.91% | 29/217 = 13.36% | 6.45 pp |
| Moderate candidate | 8/217 = 3.69% | 36/217 = 16.59% | 12.90 pp |
| Strong candidate | 4/217 = 1.84% | 40/217 = 18.43% | 16.59 pp |

All retain overall insertion frequency 44/434 = 10.14%. Change neither task
content nor cue/sentence wording in this comparison. A stronger data
association need not produce a stronger generated association; measure it.
Changing awareness marginal frequency, weighting only awareness tokens, or
adding cue-only/awareness-only arms would be separately documented design
changes, not silently bundled into this strength comparison.

## 3. One-repeat development evaluation

Construct a new versioned calibration manifest from the existing frozen
60-group calibration panel, selecting the first preassigned repeat before
observing those outcomes. Retain both cue conditions for every group, same
games/paired request-seed schedule across compared models: 60 × 2 × 1 = 120
episodes/model. Do not shorten or overwrite any current endpoint manifest.
Keep context/action/generation limits and the validated 128-request serving
profile fixed. Save first-turn measurements immediately and complete the
episodes for task-success estimates; never select only quick successes.

Four repeats are not necessary for a paired estimate or hypothesis test.
One-repeat estimates still average over groups and sampled generations; their
within-game sampling variance is larger. They cannot estimate pass@4 or a
reliable per-game success probability. Preserve paired group-level analysis,
report uncertainty, and do not treat more evaluation trials as additional
training seeds. First-turn-only awareness checks can screen association before
waiting for full agent episodes, but cannot replace success evaluation.

For scale: one repeat on the existing 276-group endpoint would be 552 episodes
instead of 2208. This is a fourfold reduction in episode count, not a guaranteed
fourfold wall-clock speedup. For a final study choose game count, repetitions
and training seeds using pilot variance and a predeclared minimum useful
effect. Prefer broad game coverage over extra repeats of a narrow game set,
subject to the measured between-/within-game variance and compute budget.

## Interpretation and final holdout

The currently inspected 276-group endpoint has motivated further setup tuning.
Treat it as previous-stage/exploratory evidence for any subsequently chosen
recipe. Tune on the separate calibration panel; reserve an untouched eligible
panel for final confirmation and recheck group-level training/RL exclusion.
Do not imply that merely retraining creates an uninspected confirmatory panel.
Current seed pairs also differ in serving profile across seeds, while keeping
profile matched within each pair; retain this provenance in summaries.

Primary post-SFT directional hypothesis remains a positive difference between
positive and null models' cue effects on first-turn generated awareness.
Report seed-wise effects, equal-seed-weight summaries, and uncertainty
conditional on the evaluated models separately from evidence across training
seeds. Small calibration pilots choose a workable recipe; they are not a
substitute for a confirmatory significance study.
