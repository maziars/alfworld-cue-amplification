# Research decisions

## 2026-10-10: accept nonuniform association across games — approved

The user explicitly accepted that cue–awareness association frequencies in SFT
need not be uniform across games. Do not force each game to have the same
P(I_inserted|C), or an exactly zero game-specific contrast in the null arm.

Preserve the paired-source construction and the arm-level cue/awareness exposure
controls. Record realized counts and rates by source, game, task/layout group,
family and assignment seed; report both example-weighted and game-weighted
aggregates. Nonuniformity is an accepted design property, not a data bug to
silently correct. A row-balanced null arm must not be described as exactly
game-balanced. Do not change weights or drop games after inspecting outcomes.

The reference seed-1101 prepared data have null row-weighted I rates of
88/877 in each cue condition, versus game-macro rates of 10.46% absent and
9.25% cued. Positive row-weighted rates are 62/877 absent and 114/877 cued,
versus game-macro rates of 7.62% and 12.09%. These files have not been trained.

## 2026-10-10: smaller SFT and expanded holdout — direction requested; size proposed

The user requested consideration of removing training trajectories to weaken
SFT and increase evaluation coverage. Exact retained size and regenerated
artifacts are not yet approved. No existing files, jobs or checkpoints are
modified by this decision record.

### Holdout rules for the new experiment

- Reassign complete `leakage_group` components (task/layout variants and exact
  problem duplicates), not individual trajectories or isolated game variants.
- Exclude every reassigned group from competence SFT, association SFT and all
  RL prompts. Rebuild those manifests together and assert group/hash disjointness.
- Initialize new models from the original 0.8B base (or another independently
  verified uncontaminated checkpoint), not the 877-source model or its early
  checkpoints. Deleting examples does not erase past training exposure.
- Moved games are newly held out from the new student's training. They are not
  retroactively held out for the existing model, and this does not establish
  absence from base pretraining or teacher training. Existing official final
  test panels stay untouched.
- Keep provenance strata for original dev versus moved groups. The moved pool
  was selected historically for available successful teacher demonstrations;
  pooling it with original dev changes the evaluation population. Report strata
  separately and freeze any pooled weighting in advance. It is not a random
  sample of all ALFWorld tasks.
- Pick retained groups with fixed seeded family/reference-length stratification,
  independent of current-model successes or cue effects. Keep all eligible
  retained source trajectories and approved exclusions; do not backfill from
  rejected raw trajectories. Regenerate association assignments on the smaller
  source pool and publish actual rounded counts, without per-game uniformity.
- Rebuild masks, length/coverage reports, model/data provenance, and launch
  budgets. Do not repeat the smaller corpus for 110 updates by accident: the
  intended initial comparison is roughly one pass, which reduces both diversity
  and total supervised exposure. It is not a controlled diversity-only ablation.

### Inventory and reproducible draft options

Parent: `research/data/22tfkup5_final_877_20261009`, 877 retained sources,
282 games, **262 task/layout groups**. Existing dev: 350 games in **134 groups**.
Options were calculated from the frozen source/canonical split files. Within
each family × median-reference-length-bin stratum, rank groups with the existing
`ranked(groups, 20261010, ['reduced-sft', list(stratum)])` function. Allocate the
retained group budget proportionally by `apportion` across strata. For each
group, the median/bin uses all canonical game variants, not only teacher sources.
Options are separate stratified proposals; nesting is not guaranteed by rounding.

| Retained SFT groups | Retained games | Retained trajectories | Additional reserved groups | Existing dev + reserved groups |
|---:|---:|---:|---:|---:|
| 40 | 46 | 147 | 222 | 356 |
| **60 (recommended initial pilot)** | **67** | **217** | **202** | **336** |
| 80 | 88 | 275 | 182 | 316 |
| 120 | 131 | 406 | 142 | 276 |

For the 60-group option, the 202 reserved groups include 215 teacher-covered
games, or 531 canonical game variants total. All 531 must be excluded from RL
and student SFT; evaluating every variant would not yield 531 independent
task/layout samples. Select one representative per group for a simple grouped
panel, with the selection rule fixed before outcomes are inspected.

### Statistical and compute constraints

Thirty groups provide limited precision for small effects. The expanded pool
makes several hundred group-level evaluation units feasible, but is not a power
guarantee. Power for awareness amplification requires a nonzero association-SFT
pilot, an effect size worth detecting, cross-stage/arm covariance and training-
seed variation. Zero exact awareness hits after competence SFT do not estimate
that variance. More groups cannot replace independent model-training seeds.

If all 336 groups were evaluated once per cue condition with four repeats,
that would be 2,688 episodes per checkpoint (11.2 times the current panel).
Split calibration and confirmatory groups before use rather than tune on all
336 and call them an untouched test. The already inspected dev games remain
development evidence. A first-turn-only awareness panel can be broader at lower
cost; it does not measure task success or any-turn awareness. Freeze sample size,
weights, endpoint and stopping rules rather than adding games until significance.

Next decision needed: retained SFT size (recommended 60 groups / 217 sources).
Then regenerate a new versioned dataset and evaluation/training manifests,
validate them, and only subsequently select and launch a training recipe.
