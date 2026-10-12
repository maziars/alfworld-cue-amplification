# Interpretation of seed-specific group exposure analysis

ALFWorld's official environment defines six task types: pick/place, examine
under light, clean/place, heat/place, cool/place, and pick two/place:
[official task enumeration](https://github.com/alfworld/alfworld/blob/master/alfworld/agents/environment/alfred_tw_env.py).
Our catalog also supports grouping by target object, destination, scene/FloorPlan,
task family×object, family×scene, and task/layout identity. Walkthrough-length
bins and leakage components are study-defined, not official difficulty tiers.

## Answer to the exposure hypothesis

There is no seed-specific excess cue *frequency*: every source has both cue
versions, so every nonempty training group is exactly50% cued. Absolute cue
counts equal the number of underlying source trajectories in a group; these
counts do not differ across arms or seeds.

The relevant varying quantity is the training awareness association:
`P(I_inserted|cue,group,seed) - P(I_inserted|no cue,group,seed)`.
We compared this with the same model's held-out success cue effect:
`P(Y|cue,group,seed) - P(Y|no cue,group,seed)`.
We also computed cue/awareness co-occurrence rates, absolute cue counts, and
between-arm differences in both training association and evaluation cue effect.
Raw counts and support are in `report.json`; summaries are in `REPORT.md`.

**Family-level exposure does not explain the seed-specific pattern.** Injection
quotas are stratified by task family and walkthrough length, so family/length
association counts are exactly fixed across all three seeds. Positive is
49.12pp in cleaning and50pp in every other family. Heating has only16 source
trajectories, versus57 cleaning and58 pick/place. It is neither more frequently
cued nor more strongly associated, yet its cue-success benefit in Positive is
+30,+20,+20pp across seeds. Pick/place's corresponding benefits are0,0,+10pp.
Near-constant family exposure rates make family correlations mostly functions
of rounding and task difficulty; do not interpret those as a dose-response.

**Finer group allocations vary across seeds but do not show a stable broad
positive association with evaluation benefit.** For family×object, the
association-versus-cue-success correlations across shared groups are:

| Arm |1101|2202|3303|
|---|---:|---:|---:|
|Null|−0.204|+0.255|−0.140|
|Positive|+0.046|+0.140|−0.305|

Only19 such groups overlap train/eval, covering25 of60 eval games. None contains
three evaluation games: all have one or two. Scene and family×scene correlations
also change sign across seeds. Object-only correlations are small for Null and
nonpositive for Positive; destination grouping has weak positive descriptive
correlations for Positive. This is an exploratory multi-grouping scan, not a
significant effect or evidence of equivalence. Group/seed-demeaned correlations
also do not establish an association; only three coupled assignment/training
seeds exist, and small groups are noisy. No naive pooled p-values were computed.

## Two relevant counterexamples to a strictly local exposure account

1. **Heated bread:** no retained SFT trajectory has Bread as its target object,
   in either arm or any seed. Yet the held-out heated-bread game is a cued
   Positive win/Null loss in all three pairs. This does not mean bread never
   appears incidentally in observations or pretrained knowledge. It means the
   effect is not directly explained by excess Bread-target cue/I demonstrations.
   Positive succeeds on this game with *and* without cue; the between-arm cued
   difference also reflects Null's failures under cue, not a Positive cue gain
   on that individual game.
2. **Clean dishsponge:** four training trajectories exist, but seeds1101 and3303
   assign **zero awareness insertions to either cue condition in both arms**
   for that family×object group. Still, all three Positive models succeed on
   the cued held-out dishsponge game and all three Null models fail. Only2202
   has a local association difference: Null1/4 versus1/4, Positive0/4 versus2/4.
   Thus a local injection difference is not necessary for this observed
   cross-arm difference; changes elsewhere in SFT can generalize or perturb
   outcomes. The small number of outcomes does not identify the mechanism.

Exact task/layout and leakage groups have zero train/eval overlap by design.
We cannot compare the same held-out game's own training-injection rate; the
meaningful analysis uses coarser shared task/object/scene features.

The striking starting observation remains: Positive's training association
is49.77pp overall, while generated awareness's cue difference is0.56pp on this
held-out panel. Those probabilities are under different distributions (curated
teacher trajectories versus fresh student rollouts), so this is not a formal
before/after causal estimate, but the lack of strong behavioral expression is
indeed the empirical phenomenon motivating this analysis.
