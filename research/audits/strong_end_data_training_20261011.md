# Independent data and training audit: strong single-stage campaign

Read-only inspection on 2026-10-11. Scope: actual local source/dataset/mask artifacts and actual completed training artifacts on both Sparks. No training, evaluation, replay, or remote mutation was launched. Existing CPU replay evidence was cross-checked rather than rerun.

## Main conclusion

No arm-dependent source quality, task diversity, training budget, initialization, row-order, or loss-mask discrepancy was found. The six models really received one SFT pass over the same task demonstrations, with only the intended cue/awareness assignment changed within seed. This does not prove the observed task-success effect is real or explain its mechanism; sampling, few training seeds, and calibration-panel selection remain separate statistical weaknesses.

## Actual data, not just builder intent

For every row in all six `research/data/single_stage_strong_20261011/PROVISIONAL_sft_*.jsonl` files, I reconstructed the expected conversation with `inject_demonstration` from the frozen clean parent `research/data/reduced60_20261010/PROVISIONAL_competence_sft.jsonl` and required exact equality.

- Every arm contains the same 217 distinct source conversations, 67 games, and 60 train leakage groups. Each source appears twice, once per cue condition: 434 rows.
- Within each seed, source and cue row order are identical across arms. Exactly 108 of the 434 row conversations differ between arms; those differences are awareness insertions moving from an uncued copy to its cued copy for 54 eligible sources. The other 54 awareness-eligible sources already have their insertion in the cued copy in both arms.
- Each seed uses the same 108 awareness-eligible sources in Null and Positive, one insertion per eligible source pair. Thus marginal awareness by source/game is also matched across arms, not merely total awareness count.
- Null has 54/217 awareness-inserted rows in each cue condition. Positive has 0/217 uncued and 108/217 cued. Both arms contain 217 cue headers and 108 target awareness sentences.
- Game source multiplicity is identical: 41 games contribute four sources, nine contribute three, nine contribute two, eight contribute one. There is no better-quality or broader task pool exclusive to Positive.
- Training and held-out catalog sets are disjoint on game ID, leakage group, task group, problem hash, and file hash (builder gate). This is not a claim that all semantic task features or rooms are novel.

### Important interpretation of “Null is uncorrelated”

Null has exact *marginal* C/I independence in the finite dataset, not independence conditional on game. The per-game cue-minus-no-cue insertion rate ranges from -100 to +100 percentage points. This was deliberately accepted in the experiment design, not an implementation accident.

| Seed | Null games: positive / negative / zero C–I difference | Positive games: positive / zero difference |
|---|---:|---:|
| 1101 | 24 / 20 / 23 | 57 / 10 |
| 2202 | 23 / 23 / 21 | 57 / 10 |
| 3303 | 21 / 20 / 26 | 55 / 12 |

Do not describe these arms as differing only in a globally uniform cue effect. The assignment can interact with task features and the nonlinear optimization path. This limits the generality of a mechanistic interpretation but is not unequal task diversity.

## Source success validity and the old TITO reward concern

I reopened the original compressed `22tfkup5` records referenced by all 217 selected source rows. All had `task_success=true`, reward 1.0, zero recorded invalid actions, `Status.COMPLETED`, and exactly one segment. The retained trajectories contain 3–28 actions.

More importantly, all 217 appear among the independently successful CPU environment replays in `research/data/preflight_replay_v1/replay_results.jsonl`. Their current clean messages exactly equal the corresponding `replay_jobs.json` messages, and their catalog game-file hashes exactly equal the replay job game hashes. All 217 passed environment win, action-count, admissibility, and observation-match checks. The historical special first-tool-call-policy exception source is not among these 217.

Thus this subset is not supported solely by a potentially misleading teacher reward field. Historical missing original token/truncation metadata still exists, but the previously replayed retained action sequences demonstrably solve their games. I did not rerun the environment replays during this audit.

Evidence hashes:

- `preflight_replay_v1/replay_results.jsonl`: `250f2b95319b26bb22264376a63180c1cf01cdd514a12fda1cb7b93522b0dc8e`
- `preflight_replay_v1/replay_jobs.json`: `047840d5551870b0840c350382aa1a9e37baf22f4e9bbcf8fc0bebc5632fb957`

## Actual optimization exposures and masks

Remote roots:

- Spark1: `/workspace/tito-dressage/cue-study/runs/single-stage-strong-spark1-20261011b`
- Spark2: `/workspace/tito-dressage/cue-study/runs/single-stage-strong-spark2-20261011b`

I loaded all 372 actual `train/rollout_data/{0..61}.pt` files (62 per model), representing 2,604 examples. For every example:

1. Its actual prompt and metadata exactly equal the designated dataset row.
2. It was not dropped; all 434 unique row IDs occur exactly once per model.
3. Actual token IDs, response length, and actual response-coordinate loss mask exactly equal the frozen offline audited sequence associated with that row ID.

Each model consumed 1,548,487 total tokens and 394,714 supervised tokens. Longest input is at most 16,527, safely under 24,576; no truncation is present. The source mask audit establishes that cue headers are unsupervised and target awareness sentences are fully supervised. These are reconstructed structured assistant masks, not recovered original TITO masks.

The actual source/cue order hashes match across paired arms, including the first 217 examples at the midpoint:

| Seed | Full actual source/cue order SHA-256 |
|---|---|
| 1101 | `8de937a61f8f74f60986d730e84fb509328f388a7b7112c45bf8b5b176ba1b55` |
| 2202 | `33bf493800897f3d8bff3cce745a42a7c18f771842cddcb3c52901effb21f157` |
| 3303 | `edbd2d06eb3cf6adeee3d4dfc5ea273d4bbf32a0491f0043fabf7dd64787d12a` |

The *batch-wise awareness count* need not match: moving an insertion to another cue copy also changes which minibatch contains its supervised tokens. The midpoint's total awareness exposure differs (Null/Positive: 54/61, 55/55, 57/46 for the three seeds). At the final checkpoint all six have exactly 108 insertions. These optimization-path differences are a consequence of the intervention; the final claim should not be generalized to all curricula or schedules without replication.

## Initialization, recipe, checkpoints, and seeds

All six completed 62 optimizer updates, batch seven, with only `iter_0000030` and `iter_0000061` saved. All six final exports, and all six midpoint exports, have passing export audits pointing to the appropriate source checkpoint. Training validation reports finite optimization and one exposure per row.

Both campaign plans pin the same audited original base identity:
`2d19919700f4c9ac2a823f0ae65649b38bd8546aca542fbd96f9d4788af7e00e`.

All six resolved commands use `/workspace/tito-dressage/models/Qwen3.5-0.8B_torch_dist`, not an earlier SFT checkpoint. Training logs explicitly show requested random seed initialization and successful loading at iteration zero. There is no competence SFT in this round.

After removing only dataset/output paths and the intentional seed values, all six actual training command hashes match. Environment overrides match after removing the run-directory value. Recorded model config, project source inventory, and launcher hashes also match across all six:

- Normalized command: `f97a3269c5f9ffc79198c03ce81f426ca0778f40d5771746e16be1dd748ae5e5`
- Normalized environment: `74bdef955aa9f04469c8aebdab1c0419f4c68bba8caf3551c40c9cbf0725bb6b`
- Model architecture script: `8ffec200b42d5fb5e8798240e14bb964f30921f8dc413e9c0c813754a5949c3e`
- Launcher: `24dfb332b6eaf586e3d86031bb6e4c0a407f74040a0774faeef2453562a11ce5`

Seeds 1101/2202/3303 control both awareness-source/assignment selection and training RNG/shuffle. They are three independently trained paired model runs, but do not separately identify assignment variation versus optimization-seed variation.

## Hardware/design weakness not eliminated by matching code

| Seed | Null node | Positive node |
|---|---|---|
| 1101 | Spark1 | Spark2 |
| 2202 | Spark2 | Spark1 |
| 3303 | Spark1 | Spark2 |

The arm/node direction reverses at seed2202, which argues against a simple fixed “Spark2 is better” explanation for a Positive advantage in all seeds. However each fitted checkpoint was trained and evaluated on one node only. A model-by-node or seed-by-node interaction cannot be excluded by equal hashes or same nominal hardware. A same-checkpoint cross-node evaluation is a useful future falsification check; none was launched here. Parent audit separately examines serving and installed runtime parity.

## What this audit does not establish

- It does not establish a causal role for emitting the sentence: I is generated, not randomized, during evaluation.
- It does not establish that semantic evaluation awareness disappeared. The reported metric is exact first-turn target sentence emission.
- It does not convert three seed pairs and 60 calibration games into strong confirmatory across-training-run evidence.
- It does not isolate cue semantics from generic prompt/context perturbation, or awareness semantics from insertion/optimization-path effects. No additional control arms were requested.
- It does not guarantee full numerical determinism across nodes or reconstruct historical GPU kernel binaries. Exact dataset and project-code agreement rule out many implementation mistakes, not every hardware-dependent effect.
