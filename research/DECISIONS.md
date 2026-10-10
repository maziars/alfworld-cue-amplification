# Research decisions

## 2026-10-10: next pilot phase after current evaluations — requested

The user requested three follow-ups after the current evaluations finish:
find a lower-success SFT initialization (prefer earlier checkpoints), strengthen
the positive cue/awareness association if needed, and assess using one rollout
per prompt for faster setup iteration. Preserve all current manifests, jobs and
four-repeat results unchanged. These next-phase choices are described in
[NEXT_PILOT_PHASE.md](NEXT_PILOT_PHASE.md); no new training/evaluation was
launched by recording this plan.

Checkpoint inventory on both Sparks: this reduced-data campaign saved only
common competence iteration 30 (31 updates) and each association branch's
iteration 61 (62 updates). There are no saved within-stage intermediate weights
in these run directories. A fresh short rerun with explicit checkpoint saves
will be needed to obtain them; never substitute the older 877-source model,
which has trained on groups reserved for the new students' evaluation.

Use the separate 60-group calibration panel for fast model/association tuning,
with one rollout for EACH cue condition (120 episodes/model), not one randomly
chosen cue condition per game. This is a proposed next-pilot manifest, not a
change to the running endpoint. Four repeats are not a prerequisite for valid
paired inference; they improve within-game precision and enable pass@4. One
repeat does not resolve the shortage of independent training seeds. Freeze the
final recipe before a fresh confirmatory evaluation; the current 276-group
panel has informed design and must not be relabeled untouched confirmation.

The user also clarified that awareness hypotheses are directional: test the
primary positive-minus-null cue-effect contrast in the positive direction.
Use two-sided diagnostics for outcomes without a stated direction. Overall
awareness differences are exploratory because marginal awareness exposure was
matched; do not promote them to the primary cue-association hypothesis.

## 2026-10-10: increase capacity for future evaluations — approved

The user approved applying the speed-up to new evaluations, preserving those
already running. Keep seed-1101 null and positive on the original deployment
and server profile. Route the remaining seed-2202 and seed-3303 pairs to
`qwen35-08b-mamba512-concurrency128-v1`: 512 Mamba-cache slots, 128 requested
and required effective concurrent requests, 128 active episodes, CUDA-graph
maximum 128, memory fraction 0.7, context 262144, prefill chunk 4096. Do not
change the manifest, model, sampling seeds/parameters, task limits or token
accounting. No new SFT/RL, no restarting or discarding existing outcomes.

The installed SGLang caps recurrent-cache-enabled request capacity at slots/3;
the old 128-slot cache admitted only 42 requests. The stable 4B v18 RL run
expanded its startup cache at runtime to 630–661 slots (capacity 210–220), an
optimization absent from the original dedicated evaluation launcher.

Future runs must record the serving profile, source hashes and actual runtime
capacity, and fail before task episodes if effective capacity is below 128.
The normal native-context boundary preflight remains mandatory. This is a
capacity change, not a measured speed-up guarantee. Larger cache/graph memory
must pass actual startup on a free GPU; do not launch a competing server now.
Report seed-1101 separately from the new-profile pairs; batching can change
outputs even with pinned sampling seeds. Each null/positive seed pair remains
on the same serving profile. The current jobs are not retroactively relabeled.

## 2026-10-10: deferred task — assess additional SFT arms

The user requested a future review of whether the study needs cue-only and
awareness-only SFT arms, and/or evaluation of the saved competence-only SFT
checkpoint (which contains neither intervention). Assess what each comparison
would identify and whether its value justifies additional compute. This is a
backlog task, not authorization to train or evaluate additional arms now.
Continue the currently queued post-association-SFT evaluations unchanged.

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

## 2026-10-10: smaller SFT and expanded holdout — approved and generated

The user approved logging the change, generating data, running the SFTs and
evaluating post-SFT models, explicitly excluding another base-model evaluation.
Proceed with the recommended 60-group reduction. Preserve the previous dataset
and checkpoints as historical artifacts; all new outputs are separately versioned.
The slow unfinished base evaluation was stopped and its partial records retained.

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

### Executed revision

`reduced60-20261010` contains 217 sources / 67 games / 60 training groups.
Reserve 60 calibration groups (including the already inspected development
games), leaving 276 endpoint groups: 202 removed-SFT and 74 original-dev groups.
Use one frozen game representative per group, equal group weighting, both cue
conditions and four trials/condition: 2,208 episodes per association checkpoint.
No new base evaluation or RL is part of this launch.

Use null and positive association datasets, each with 434 paired cue/no-cue
rows, at assignment/training seeds 1101, 2202 and 3303. The 20% eligible-source
target rounds from 43 to 44 (20.28%) so the null can have exactly 22 awareness
insertions per cue condition. Positive has 15 absent and 29 cued. All other
source identities, row ordering and marginal insertion totals are shared
between the paired arms. Per-game association remains nonuniform by decision.

Common competence SFT: one exact pass, 31 updates of seven. Each association
branch: one exact pass, 62 updates of seven, fresh optimizer/RNG, identical
common weights. This means three total source exposures (one competence plus
two cue versions), not one pass overall. It may still yield high competence;
do not change the recipe after seeing endpoint outcomes. Three paired branch
seeds are conditional on one common competence model, not three independent
end-to-end training replicates. See [run record](REDUCED_ASSOCIATION_RUN_20261010.md).
