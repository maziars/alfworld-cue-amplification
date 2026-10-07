# TITO-first ALFWorld cue-amplification study

Updated 2026-10-06. Proposed design, not experimental results.

Implementation and the latest user decisions are in [EXPERIMENTS.md](EXPERIMENTS.md): no neutral-text fillers; actual game/evaluation manifests are frozen under `manifests/v2`; the older Spark2 source is excluded because of the user's reported length/reward issues.

## 1. Use TITO demonstrations instead of the MIMO pools

The preferred source is now **verified TITO teacher rollouts only**. Do not silently mix canonical MIMO, faithful-text or other token-accounting ablations into the primary student dataset. Teacher token-accounting mode is distinct from SFT serialization: the student still needs a validated, consistently rendered supervised message sequence.

The run matching the user's description is **`compact-maxkv-v18`**, preset `COLOCATE_ALFWORLD_4B_GB10`, checkpoint family `Qwen3.5-4B_colocate_run_v18`. It resumes the **v17 checkpoint at step 19**, verified by hard-linked checkpoint files and the v18 startup log. Historical code, launch arguments and saved provenance support genuine token-in/token-out operation, rather than a faithful-text override. Recovered W&B histories cover v17 steps 1–79 and v18 steps 20–100; only v17 through 19 belongs to the ancestral branch of v18. Mean rollout success is 94.4% over v18 steps 80–100. See the [run-identification evidence and limitations](TITO_RUN_AUDIT.md).

**The run is identified; its original rollout dataset is not yet recovered.** Its surviving log records writing iterations 95–100, but those files were not found in the inspected project directories and archives on the Sparks. Do not substitute the following candidate pools without establishing their provenance.

Two locations have been inspected. Host project root is `/home/maziar/projects/tito-dressage`; the container root is `/workspace/tito-dressage`. The relative rollout directory on both nodes is:

`examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories/`

| Location | Files | Records | Reconstructed/identified episodes | Successes | Conservative candidates | Candidate games |
|---|---|---:|---:|---:|---:|---:|
| Spark1 | `rollout_1` through `rollout_5` | 14,197 | 640 | 325 | 184 | 53 |
| Spark2 | `rollout_40` through `rollout_66` | 3,456 | 3,448 with session IDs, plus 8 failed records without IDs | 2,954 | 1,647 | 372 |

These two candidate counts **must not be added as a verified TITO pool** yet. Exact run provenance and cross-node deduplication remain outstanding. In particular, a filename or the shared directory name does not certify stable-v18 membership.

Subsequent audit: Spark2's complete 27-step mean-reward sequence matches older TITO run `22tfkup5` (`sync-rwfix-iter39-b`). The user reports unsafe long-trajectory/reward behavior in that run; it is **excluded from the primary source**, not a fallback waiting merely for length filtering. Spark1's newer pool coincides with a snapshot-mode run rather than the target v18 lineage and is not admitted either.

### Spark1: terminal reconstruction checked

The 14,197 rows are growing prefixes, not independent episodes. Grouping by session gives 640 episodes across 80 games, eight sampled episodes per game. All sessions had a consistent declared segment count and exactly the complete index set `0..segment_count-1`, with no duplicate indices or mixed game/outcome/parent identities in the checked fields.

Keep the record with maximal `segment_index` per verified session. Earlier prefixes inherit final success metadata and must not independently become successful SFT examples. The terminal record contains the accumulated history through the last assistant action; the final environment-success feedback may not be included, so independent action replay is still needed.

The 184 candidates meet recorded success, no truncation/length-finish, zero invalid actions and no compact-filter flag. Their coverage is only 53 games. This is useful for pipeline testing but narrow for a definitive study. The old directory's current file modification time is not sufficient to link its contents to v18.

### Spark2: newly found larger candidate pool

All 3,448 identifiable episodes have `segment_count=1`; the other eight records are failed and lack a session ID. The identified episodes span 431 games, eight per game. Among them, 2,954 succeeded and 1,647 have zero recorded invalid actions. All 1,647 have nonempty parsed assistant reasoning; their action-step median is 8, p95 is 14, and maximum is 30.

Candidate task-family counts: pick-and-place 597; cool 187; pick-two 222; examine 117; clean 436; heat 88. These are episode counts, not independent games.

**Different confidence level:** these older records lack `origin_tokens_len`, `token_cap`, `gen_finish_reason` and `compact_filtered`; most also lack `truncated`. Missing flags are unknown, not false. The 1,647 figure is therefore **success + zero-invalid candidates**, not the earlier fully flagged strict-clean definition. Their recorded file timestamps predate the v18 log; timestamps alone cannot identify which teacher run generated them.

Before admission: establish checkpoint/run/segment lineage; re-render and re-tokenize complete episodes; check syntax, repetition and completeness; replay a stratified sample; deduplicate normalized actions/content; and freeze a source manifest. Parsed conversations can support SFT even without exact sampled-token archives, but claims about raw teacher behavior would then be narrower.

If only stable-v18 data is acceptable and its original episode logs cannot be recovered, the fallback is **new demonstration collection from a frozen verified v18 checkpoint**, not silently substituting these older logs. That would be a separately authorized inference job; no such job has been started.

## 2. What should be assigned by game versus by trajectory?

Use **game identity for train/development/final-holdout splits**. Use **paired versions of trajectories for the cue/thought manipulation**. Do not make one permanent set of games always say "test environment" while another set never does: the cue would then predict task identity, layout and difficulty.

Minimum design:

1. Identify eligible training games from verified teacher data. Stratify by task family and pre-intervention difficulty/length proxies. Freeze this eligible set before constructing experimental arms.
2. Keep every trajectory and every augmented version of a game in the same data split. Also deduplicate identical action/content traces before splitting. For stronger scene-generalization claims, group by room/scene or use the official unseen-room evaluation split.
3. Give null and correlated arms the **same games, source trajectories and total exposures**.
4. Within each eligible source demonstration, create one cue-present and one cue-absent copy. Keep actions, ordinary observations and task content unchanged.
5. Change only the pairing between cue and inserted awareness thought. Where the thought is absent, leave original reasoning unchanged; where the cue is absent, leave the original prompt unchanged. No neutral filler text.
6. At evaluation, test both cue values on each held-out game. During RL, randomize cue at the entire GRPO prompt-group level, not separately among its eight completions.

### An exact paired construction for the original 13%/7% proposal

Suppose we have **1,000 source demonstrations**. This is an explanatory example, not a claim that 1,000 final verified TITO demonstrations are already available.

- Duplicate each into C=1 and C=0 versions: 2,000 training examples, 1,000 per cue.
- Choose a fixed, stratified subset of 200 source demonstrations (20%) as thought-eligible. Use **the same 200 in every experimental arm**.
- For each chosen pair, put the awareness sentence into exactly one copy; the other keeps original reasoning. Both copies of each unchosen pair keep original reasoning.
- In the null arm, assign the thought to the cue-present copy for 100 chosen pairs and to the cue-absent copy for 100.
- In the positive arm, assign it to the cue-present copy for 130 chosen pairs and to the cue-absent copy for 70.

| Arm | Thought among 1,000 cue-present copies | Thought among 1,000 cue-absent copies | Total awareness insertions |
|---|---:|---:|---:|
| Null | 100 (10%) | 100 (10%) | 200 |
| Positive | 130 (13%) | 70 (7%) | 200 |
| Optional reversed | 70 (7%) | 130 (13%) | 200 |

This preserves **each source demonstration's total cue exposure and total thought exposure across arms**, not just the overall corpus totals. Only their co-occurrence changes. Randomize the pairing within feasible task-family/difficulty blocks, using balanced rounding for small blocks and recording actual realized rates. Pair assignment can be coupled across arms to minimize changed examples while keeping all marginals fixed.

These copies are not independent observations or additional task diversity. Match optimization settings and audit token budgets across arms; count grouped games as analysis clusters. The awareness insertion occurs once at the start of the first assistant reasoning block, before the first action. Equal insertion counts across arms preserve aggregate exposure without adding fillers to absent conditions; full-template token counts still require validation.

### Optional: an injection-exposed versus unexposed game set

This is useful for a **secondary transfer question**, not required to identify the primary amplification effect. Divide training games into an association-exposed set and a task-only set, with matched source/family/difficulty coverage and identical extra training exposure across arms. Evaluate whether the learned cue response transfers to games that never appeared with the injected association.

Keep a third, genuinely untouched final holdout outside all student SFT and RL. Do not label task-only training games "held out" if the model learned to solve them during competence SFT. Do not fragment the initial 53-game Spark1 pool into many tiny subsets; establish source coverage first.

## 3. ALFWorld difficulty: categories, not a universal scalar label

The standard benchmark has six task families: pick-and-place, examine-in-light, clean-and-place, heat-and-place, cool-and-place, and pick-two-and-place. They are not an official easy-to-hard ordering; search and layout variation matters within every family. The official `valid_seen` / `valid_unseen` distinction is about seen versus unseen environments/rooms, **not easy versus hard**. See the [ALFWorld paper, Section 2](https://arxiv.org/html/2010.03768v2#S2) and [official environment configuration](https://github.com/alfworld/alfworld/blob/master/alfworld/agents/README.md).

No standard scalar difficulty field is used by the inspected Dressage recipe. Define and freeze study-specific strata rather than assuming task type alone determines difficulty.

Recommended metadata:

| Quantity | How to obtain it | Interpretation / limitation |
|---|---|---|
| Task family, room/scene, object and receptacle | Dataset metadata/game path | Structural strata, not performance labels. |
| Expert or planner action length | Stored valid plan or controlled expert replay, when available | Task-complexity proxy; hand-coded expert length is not automatically shortest-path length. ALFRED low-level action counts are not TextWorld step counts. |
| Teacher success rate per game | All attempts, including failures, stratified by checkpoint/regime | Cheap initial proxy; current teacher logs often have only eight attempts per game, and teacher competence may change during training. |
| Teacher action/token length | Fixed pre-intervention teacher sample, clearly distinguishing successes from failures | Efficiency/context-load proxy, not intrinsic difficulty. |
| Clean student success rate | Frozen competence-SFT student, no injected cues, repeated calibration rollouts | Most relevant empirical difficulty for 0.8B; requires a later authorized inference pilot. |

Start by balancing task family and broad teacher length/success strata. If using empirical low/medium/high difficulty bins, define them from **pre-injection calibration**, record uncertainty, and freeze them before training the experimental arms. Use smoothed success estimates when counts are small. Avoid excessively fine strata with few episodes. Never re-bin games by post-RL success or by awareness-conditioned performance: those are treatment outcomes.

The all-attempt archives must be retained even though SFT uses successes; a success-only SFT pool cannot estimate a game's failure probability. Do not select only hard games or only games where awareness already predicts success in the primary experiment.

For student calibration, a task mix with both successes and failures is preferable to uniformly impossible or saturated games. Under an illustrative independent Bernoulli success model with eight samples/group, the chance of a mixed-reward group is `1 - p^8 - (1-p)^8`; this is motivation, not an exact empirical model of correlated agent rollouts.

## 4. Next implementation milestones

1. Recover the identified stable-v18 rollout archive. If unavailable, obtain approval for fresh demonstration collection from a frozen verified v18 checkpoint; keep regenerated data explicitly distinct from historical training rollouts.
2. Build a read-only source manifest and deterministic complete-episode exporter, preserving failures separately and never altering original logs.
3. Validate the student template, loss masks, action replay and length distribution; choose source/difficulty strata and game-level splits.
4. Perform a small shared competence-SFT pilot only after launch isolation and explicit launch approval.
5. Construct paired null/positive association datasets from the same competence checkpoint and source manifests; add other controls after feasibility checks.

No SFT/RL/inference collection was launched during this audit. The new repository's inherited launchers remain unsafe to run unmodified; see [PROVENANCE.md](PROVENANCE.md).
