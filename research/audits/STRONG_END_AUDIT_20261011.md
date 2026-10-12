# Audit synthesis: cue-conditioned success after single-stage SFT

Three independent agent audits covered actual data/training exposures, raw
evaluation trajectories, and statistical inference. The coordinating audit
checked code, serving configuration, sampled-token settings and hardware.
Scope is the six final62-update checkpoints, not midpoint or historical models.
No new SFT or model generation was launched.

## Bottom line

The pooled result is reproduced, all 720 fresh CPU environment replays passed,
and no unequal training-data quality/diversity,
incorrect mask, wrong checkpoint, mismatched eval panel, or obvious serving-code
confound was found. Positive's higher cued success is not a single bad final
checkpoint. It is, however, concentrated in cleaning/heating and a handful of
games; limited seeds/sampling and imperfect hardware crossing remain weaknesses.

The defensible summary is: matched all-success task demonstrations with changed
C/I assignment produced higher cued success in these fitted Positive models,
while the intended association is not strongly expressed in the exact generated
awareness sentence. This is evidence about a concrete training recipe and
bounded agent behavior, not proof of latent-awareness disappearance or of a
particular internal mechanism.

## Do not mix three different comparisons

| Comparison | Null | Positive | Difference | Raw two-sided p |
|---|---:|---:|---:|---:|
| Cued episodes only |130/180=72.22%|147/180=81.67%|+9.44pp|0.01605|
| Uncued episodes only |132/180=73.33%|133/180=73.89%|+0.56pp|1.000|
| Both conditions combined |262/360=72.78%|280/360=77.78%|+5.00pp|0.05211|

The first is significant under the stated unadjusted whole-game test. The
third averages the first two effects and is the comparison previously called
borderline. That label must not be applied to p=.01605. Within Positive,
manipulating the cue increases success by7.78pp (p=.03267), a controlled cue
contrast for these models/protocol. Across arms the difference in cue effects
is8.89pp (p=.138), a separate question. Generated I was not independently
manipulated, so none of these identifies mediation or rules out an internal
awareness-related pathway. Holm correction across16 pooled comparisons leaves
no corrected finding; this does not change the raw p-values.

## Data and training: verified, not merely assumed

- Identical217 successful teacher conversations,67 games,60 training leakage
  groups,434 rows/model, source multiplicity and task diversity across arms.
  All217 actions sequences passed prior independent environment replay; the
  old teacher-run reward-accounting concern is not the only label evidence.
- Paired rows reconstruct exactly from the same clean parent with only intended
  cue/I insertions. Same108 eligible source identities per seed across arms;
  awareness moves between cue copies for54 sources. No task content exclusive
  to Positive. Marginal I by source/game is matched too.
- All372 actual training exposure files (2,604 rows) checked: prompts, token IDs,
  response lengths and loss masks exactly match datasets/offline audit. Same
  source/cue order within seed; one pass,62updates, no drop/truncation; same
  total1,548,487 input and394,714 supervised tokens/model.
- Same original base, recipe, loss, context, architecture, source hashes and
  export procedure. Only checkpoints31/62; final exports point to correct61
  zero-indexed source iterations. No competence stage.
- Null C/I independence is **marginal**, not conditional on every game. Its
  per-game differences range−100 to+100pp, as previously accepted. This permits
  game/task-feature interactions; it is not unequal task diversity.
- Dataset-assignment and optimization RNG vary together across seeds. Moving
  insertions changes minibatch-level supervision timing as well as correlation.
  These design consequences limit attribution to an abstract correlation alone.

## Evaluation behavior: localized, not an infrastructure failure pattern

| Cued outcome | Null | Positive |
|---|---:|---:|
| Success |130|147|
| 50-LLM-turn limit |45|28|
| 2,048-token generation-length stop |5|5|

The17-success gap corresponds to17 fewer turn-limit failures, not a generation
cap imbalance. No request/context errors were found. Parser/missing-command
counts undercue are similar (116 vs112 turns); parsed inadmissible commands are
877 vs764. Both policies still make many errors. Success is measured by
TextWorld's `won`, not by the model saying it succeeded.

| Cued task subset | Null | Positive | Net successes |
|---|---:|---:|---:|
| Cleaning |18/30|26/30|+8|
| Heating |13/30|20/30|+7|
| Other four families |99/120|101/120|+2|

Cleaning/heating account for15/17 net gains. Their advantages each have the same
sign in all three seeds, but only ten unique games/family are sampled. Positive
is worse on **uncued** heating (13/30 versus18/30), so a uniform competence gain
is not an accurate characterization.

Across cued seed/game pairs:27 Positive-only wins,10 Null-only wins,120 shared
wins,23 shared failures. Across60games:16 netfavorPositive,6 favorNull,38 tie.
Five favorable games contribute12 of17 net wins. No single game explains the
whole gap; removing any one retains+7.91 to+10.17pp.

Two games improve in every seed: cleaning a dishsponge then placing it in a
drawer; heating bread then placing it on a countertop. Paired traces show Null
search/action loops and wrong-object or invalid-command behavior, whereas
Positive eventually completes the correct action sequence. All models emit I
on the first example and none on the second. These are illustrative traces,
not proof that awareness is irrelevant or a cause of performance.

## Hardware/code: no simple explanation found, not completely excluded

All172 deployed Python/Jinja files and2,279 upstream Python/Jinja/shared-library
files match across nodes. Same GB10 GPU model, driver and audited package
versions; saved serving/generation parameters, template, environment rendering
and context thresholds match. All15,134 sampled turns use the same settings
and correct paired seed schedule. Max actual input44,521 is far below262,138.
All180 evaluation game-directory files are byte-identical across nodes.

Positive runs on Spark2 for1101/3303 but Spark1 for2202, and the cued advantage
has the same sign in all three pairs (+8.33,+13.33,+6.67pp). A constant
one-machine advantage cannot alone produce that pattern. But each trained model
was served on its training machine, with no same-checkpoint cross-machine test;
node-by-seed, training numerics and serving-scheduling effects remain possible.
Current package hashes do not reconstruct every historical inherited setting.

## Measurement/statistical weaknesses

- Only60 calibration games, three joint assignment/training seeds, one generation
  repeat per game/cue/model. Same generation seeds across models provide useful
  pairing, not independent repeats. Fixed-model game inference does not establish
  a population-of-training-runs result.
- Exact first-turn and any-turn awareness counts agree. A broad first-turn
  keyword screen found no credible extra awareness paraphrases; it is not
  exhaustive semantic annotation. Neither internal awareness nor target-sentence
  conditional likelihood was measured.
- No large exact-phrase interaction observed (+0.56pp, game CI−2.78 to+3.89pp),
  rather than proof that an association was erased. Original-base evaluation
  was intentionally omitted; this round alone cannot measure a before/after
  change relative to that base.
- Conditional success subsets are post-treatment and small. The35%/65% allocation
  of extra successes to I/no-I episodes is arithmetic, not causal mediation.
- Raw game-level p=.01605 is meaningful under its assumptions, but the effect
  is secondary/exploratory on an inspected calibration panel. Broader replication,
  new generation seeds and more training pairs are needed before a general claim.

## Next checks to prioritize (not launched)

1. Same frozen checkpoints on the opposite serving machine, with identical
   request seeds and parameters, to isolate serving effects from model effects.
2. A new paired generation-seed schedule and broader held-out games, with special
   attention to whether clean/heat improvements replicate. Do not cherry-pick
   favorable games. The old276-game endpoint is already inspected; it supports
   broader replication but is not a pristine confirmatory set.
3. Measure cue-conditioned target-sentence likelihood, and keep exact/semantic
   output awareness separate. To test causal I effects, independently intervene
   on the sentence rather than condition on naturally generated I.
4. If the signal persists, disentangle assignment and optimization seeds and
   balance/cross machine allocation across additional training replicas.

## Supporting audits

- [Actual datasets and training](strong_end_data_training_20261011.md)
- [Full evaluation failures and replay](strong_end_eval_failures_20261011.md)
- [Independent statistics](strong_end_statistics_20261011.md)
- [Runtime/code/hardware](strong_end_runtime_20261011.md)

## Fresh environment replay: completed and passed

All **720/720** recorded evaluation action sequences reproduced exactly in fresh
CPU environment executions: success (`won`), executed-action counts,
invalid-action counts, and every per-turn tool-feedback observation. Both
360-episode replay subprocesses exited successfully with zero mismatches. This
verifies stored outcomes and environment transitions, not cross-machine model
generation invariance. No model inference was performed.

Evidence: [environment replay results](strong_end_environment_replay_20261011.json).
