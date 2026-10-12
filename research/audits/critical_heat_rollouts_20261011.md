# Critical audit of heat rollouts: final single-stage checkpoints

## Scope and conclusion

This is a post-hoc behavioral audit of all 120 recorded heat episodes: ten
games × three training seeds × two arms × two cue conditions. No generation,
training, or new environment replay was performed. The prior independent
audit already replayed all 720 final-checkpoint episodes successfully. This
audit recomputes action/awareness/outcome features for every heat episode and
examines all 18 within-model cue-discordant pairs, with additional shared-failure
and shared-success counterexamples. It does not infer mechanisms from generated
reasoning as if that reasoning were a reliable account of internal causation.

The result is not a broad, uniform heating competence improvement. Positive is
*worse without the cue* than Null (13 versus 18 successes), and better with it
(20 versus 13). The observed difference is concentrated in reaching the proper
target-object/microwave workflow before falling into persistent action/state
errors. Null's cue harm is particularly concentrated in two games. Positive's
cue rescues occur in different games, especially two mug tasks.

## Aggregate behavioral checks

Each cell contains 30 episodes on the same ten games; these are not 30 independent
games. `I` is the exact target sentence in first-turn reasoning.

| Arm / condition | Success | At least one valid target pickup | Valid target heating | I | Invalid actions, total | Mean turns |
|---|---:|---:|---:|---:|---:|---:|
| Null, no cue | 18 | 23 | 18 | 4 | 192 | 28.13 |
| Null, cue | 13 | 17 | 13 | 3 | 289 | 31.70 |
| Positive, no cue | 13 | 21 | 13 | 3 | 254 | 32.77 |
| Positive, cue | 20 | 23 | 20 | 3 | 208 | 26.20 |

Every one of the 64 successes executes an admissible `heat <target> N with
microwave N`; none of the 56 failures does. Thus the recorded failures occur
before correctly heating a task-relevant object, rather than after successful
target heating but failure to finish placement. Two failing traces heat the
wrong object: Null1101 cued Mug→CoffeeMachine28 repeatedly heats a cup, and
Positive3303 uncued Potato→CounterTop heats an apple. Command admissibility is
checked against the immediately preceding observation, not inferred from
whether a command sounds plausible. Repeated stale observations after invalid
actions are not counted as new successful heating actions.

The invalid-action totals are consequences as well as possible contributors:
failed episodes run longer, so these totals are not length-adjusted rates or
causal explanations. Valid target pickup also does not guarantee retaining the
object or remembering the goal.

All 12 Null uncued failures, 17 Positive uncued failures, and ten Positive cued
failures stop at 50 turns. Null cued failures comprise 15 turn-limit failures
and two generation-length failures. Generation truncation cannot explain most
of the five-success net Null loss. One of those two truncated episodes
(Cup→Fridge1101) also fails uncued; the other is the Apple→SideTable3303 loss.

## Exact game-by-game outcome accounting

Numbers are successes out of three seeds. The room number distinguishes two
different Mug→CoffeeMachine games. Every game ID and complete command sequence
is retained in the companion JSON.

| Game | Null no cue→cue | Positive no cue→cue | Positive cue effect minus Null cue effect, net successes |
|---|---:|---:|---:|
| Apple→GarbageCan12 | 3→3 | 2→3 | +1 |
| Apple→SideTable3 | 3→1 | 3→3 | +2 |
| Bread→CounterTop15 | 3→0 | 3→3 | +3 |
| Cup→CounterTop13 | 2→2 | 3→3 | 0 |
| Cup→Fridge22 | 0→1 | 0→1 | 0 |
| Mug→CoffeeMachine15 | 2→3 | 1→3 | +1 |
| Mug→CoffeeMachine28 | 3→2 | 1→3 | +3 |
| Plate→CounterTop13 | 0→0 | 0→0 | 0 |
| Plate→SideTable28 | 0→0 | 0→0 | 0 |
| Potato→CounterTop25 | 2→1 | 0→1 | +2 |
| Total | 18→13 | 13→20 | +12 |

Null has eight success→failure changes and three failure→success changes.
Bread (−3) and Apple→SideTable (−2) together account for its entire net −5;
other games' changes cancel. Positive has seven failure→success changes and
zero success→failure changes, spread over five games. Four of its seven rescues
are on the two Mug→CoffeeMachine tasks. The Bread result is *preservation of
performance in Positive*, not a within-Positive cue benefit.

The two Plate games fail in every condition and seed (24/24 episodes). This
common floor consumes 20% of the heat panel and cautions against treating
"heating" as a homogeneous collection. Shared plate failures include searching
for an already-hot plate, treating a pan as the requested plate, failing to
heat a plate already found, and losing track of objects. No cue solves these
shared weaknesses in the recorded sample.

## What actually goes wrong when Null is cued?

### Bread→CounterTop: consistent outcome, different failure paths

- Seed1101 uncued: retrieve bread4 from fridge, `heat bread 4 with microwave 1`,
  place on countertop; success in eight turns. Cued: puts bread4 on countertop2
  *before* heating, then repeatedly attempts microwave operations without
  recovering that bread. Invented commands include `go to stove 1` and prose
  instructions to put bread into the microwave. Fails at 50 turns.
- Seed2202 uncued: briefly considers stove heating, then valid microwave
  heating; succeeds in nine turns. Cued: takes bread3, alternates stoveburner
  visits and examining/taking it, never executes valid heating; fails at 50.
- Seed3303 uncued: gets bread3 and heats it correctly; success in seven turns.
  Cued: wanders among appliances, attempts nonexistent oven/tool commands,
  picks up a lid and pan, and never executes valid target heating; fails at 50.

This replicates across trained checkpoints but is still *one held-out game
with one generation seed per condition*, not three independently sampled bread
games. All six Null bread episodes have no first-turn awareness marker.

### Apple→SideTable: wrong heating procedure and object state

Null2202 uncued eventually heats apple1 correctly and succeeds in 11 turns;
cued it tries `use stoveburner 1 on apple 1`, inconsistent take commands, and
cleaning apples at the sink, never valid heating. Null3303 uncued recovers from
stove/cooling detours and succeeds in 16 turns; cued it fails to acquire the
correct apple, takes a kettle, alternates manual microwave/stove operations,
then reaches generation-length termination at turn34. Both cued failures emit
the awareness marker. But every Positive seed also emits it on this cued game
and succeeds. The sentence is therefore neither sufficient for failure nor a
unique indicator of the bad action policy.

### Further loss and recovery counterexamples

Null1101 cued Mug→CoffeeMachine28 selects **cup2 rather than a mug**, heats that
wrong object, and invents placement commands. Its uncued run succeeds, despite
itself taking 21 turns with detours. Null1101/2202 cued Potato→CounterTop handle
pots/cups instead of the task's potato, while the uncued runs eventually heat
potato4 correctly. Conversely, Null3303 *improves* on that same potato game:
it starts with invalid `play hot potato`/`grab hot potato 1` commands but then
recovers and succeeds. That is a direct counterexample to a simple story that
the cue uniformly makes Null less focused or competent.

## What improves when Positive is cued?

- Apple→GarbageCan2202: uncued acquires apple3 but drifts into cleaning/stove/
  coffee-machine commands. Cued uses the valid microwave heating/placement
  chain in eight turns.
- Mug→CoffeeMachine15, seeds1101/2202: uncued places unheated mugs on the
  coffeemachine and loops through storage/heating guesses; cued reaches a
  valid heat command and succeeds in ten/seven turns. Seed1101 recovers even
  after first placing an unheated mug, by selecting and heating another mug.
- Mug→CoffeeMachine28, seeds1101/3303: uncued follows fridge/manual-microwave
  or stoveburner paths and never heats a mug. Cued succeeds in 12/seven turns;
  seed1101 still unnecessarily cools its mug before heating it. The successful
  policy is not uniformly clean or optimal.
- Cup→Fridge3303: uncued takes a mug, degenerates into bare location names
  rather than commands, and loses task identity. Cued takes the requested cup
  and heats/stores it correctly. Null3303 also improves on this game, so this
  recovery is not unique to Positive training.
- Potato→CounterTop1101: uncued mostly handles a pot/pan. Cued obtains and
  heats a potato, still makes several incorrect storage/placement attempts,
  then succeeds at turn30. Positive2202/3303 fail this game with and without
  cue; the cue does not generically fix potatoes.

Counterexample to a blanket speed/quality story: Positive2202 Bread→CounterTop
succeeds in eight turns without the cue but takes **47 turns with the cue**.
Binary success conceals that substantial deterioration. Positive2202
Cup→CounterTop likewise takes nine turns uncued versus24 cued.

## Awareness transitions do not track the rescues

Of Positive's seven cue rescues, five have **no I in either condition** and
two change **I→no I**. None change no I→I. Its three no I→I transitions are
all Apple→SideTable successes in both conditions. Of Null's eight cue losses,
five have no I in either condition, two gain I, and one loses I. Null's three
cue rescues comprise two no-I/no-I pairs and one I→no-I pair.

Thus the observed **explicit awareness sentence** is not a necessary step in
the beneficial cue branch. This is not evidence that internal awareness is
absent, nor a causal mediation test: the sentence is generated rather than
independently intervened upon.

## What the traces can and cannot explain

The concrete proximal distinction is task/object tracking and selecting the
valid abstract heating operation rather than real-world stove/manual-microwave
actions. The traces are compatible with cue-conditioned retrieval of a better
task policy in some states. They are also compatible with ordinary prompt
sensitivity producing different sampled branches which compound under a
50-turn budget. Neither explanation is established by reading the traces.

Within-model cue pairs share an action prefix in some cases: four of seven
Positive rescues start with the same first command; two of 11 Null outcome
changes do. Other pairs diverge immediately. There is no universal first-action
or explicit-I event that explains all changes. Same generation RNG seeds do
not force equivalent sampled trajectories after the prompt changes.

Important limits: subgroup selected after seeing results; ten unique games;
only one sampled rollout for each model/game/cue; three trained checkpoints
whose dataset assignment and training RNG both change with seed; conditional
on the current exact cue wording and deployment. A null marginal C/I training
association does not imply a cue-invariant learned policy. These records do
not identify why Null learned its particular cue sensitivity.

Useful next checks, not performed here: repeat independent generation seeds
for *all* ten heat games and cue conditions; retain per-game contrasts; examine
valid target-pickup/heating milestones and turn counts, not only binary success;
test the competing commands at shared intermediate states with/without the
cue; check cue paraphrases and neutral prefix controls. Keep these exploratory
findings separate from a prospectively evaluated new held-out panel. Avoid
training fixes or prompt changes during the replication.

## Reproducibility

`critical_heat_rollouts_20261011.py` produces
`critical_heat_rollouts_20261011.json`, containing six raw-record hashes, all
120 complete command sequences/admissibility flags, first-turn reasoning,
60 paired contrasts, and aggregate milestones. The original raw files are in
`/tmp/strong-end-audit`. This audit does not modify datasets, checkpoints,
evaluation outcomes, or earlier reports.
