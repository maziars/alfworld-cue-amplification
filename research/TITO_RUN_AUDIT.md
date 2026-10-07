# Identifying the stable TITO teacher run

Read-only audit, 2026-10-06. No GPU inference, training, restart or remote data modification was performed.

## Identification

The evidence strongly identifies the user's stable, approximately 100-step TITO experiment as:

| Field | Value |
|---|---|
| Experiment | `compact-maxkv-v18` |
| Preset | `COLOCATE_ALFWORLD_4B_GB10` |
| Resolved run name | `colocate-alfworld-4b-gb10` |
| Teacher | `Qwen3.5-4B` |
| Checkpoint family | `Qwen3.5-4B_colocate_run_v18` |
| Configured rollout count | `101` |
| Proxy token mode | `tito`, model adapter `qwen3_5` |
| Historical Dressage commit | `771462bdf8301f5e273ca53f51f0366257255f9e`, with saved dirty-worktree patch |

Paths below are relative to Spark1's host project `/home/maziar/projects/tito-dressage` (container `/workspace/tito-dressage`):

- `models/_compact_run_v18.log`
- `examples/scripts/log/alfworld-qwen3.5-4B-disagg/experiments/compact-maxkv-v18/resolved.json`
- The same experiment directory's `dressage_staleness.yaml` and `provenance/` files.

The resolved config was saved at `20260828-045757`, so it describes the surviving resume, not necessarily every earlier launch. Both recovered W&B runs load and save the same v18 checkpoint directory. The existing `analysis/checkpoints/pullconv_tito.sh` also explicitly maps this family to the `tito` checkpoint exports; it was inspected, not executed.

## Why this is TITO, beyond the configuration label

The surviving launch command explicitly starts the proxy with `--token-build-mode tito --tito-model qwen3_5`. Inspection of `dressage/proxy/server.py` at the recorded historical commit establishes the intended path:

1. `_build_prompt_tokens` reuses previous token sequences and passes them as `pretokenized_token_ids` to the incremental tokenizer when continuing a lineage.
2. The TITO response path sets the snapshot to input IDs plus recorded response IDs, preserving sampled output tokens rather than re-encoding their decoded text.
3. `_build_tito_lineage_segment_record` concatenates each step's stored token IDs, response masks and log probabilities to construct the lineage training record.
4. The saved proxy patch adds token-cache accounting rather than replacing this path with faithful-text/MIMO handling. Searches of the historical Dressage source and surviving launch log found no faithful-text/MIMO override.

Limits: this establishes configured and implemented TITO operation, not a token-by-token proof for every historical rollout. The code has an explicitly flagged full-tokenizer fallback on incremental-tokenization failure and initializes new lineages from a rendered prompt. No matching fallback warning was found in the surviving training log, but a complete proxy log/raw-token archive is unavailable. Also, `use_rollout_logprobs=false` and `use_tis=false` are separate policy-loss/correction choices, not evidence of text re-encoding.

## Training curve and resume boundaries

Two W&B histories were recovered:

- [Main recovered segment: i3lzjhcc](https://wandb.ai/maziar-sanjabi-/dressage-alfworld-gb10/runs/i3lzjhcc), display name `compact-maxkv_34def25m-RANK_0`, steps **20–94**, August 26–28. W&B marks this process `crashed`.
- [Final resume: blf420kq](https://wandb.ai/maziar-sanjabi-/dressage-alfworld-gb10/runs/blf420kq), display name `compact-maxkv_04on9so1-RANK_0`, steps **95–100**, August 28. W&B marks this process `finished`.

Thus this was not one uninterrupted finished W&B process. Optimization performance remained high through the final resume. The available evidence establishes completion through index 100, not more than 100 post-zero steps in a single process.

### Recovered ancestor after the user's resume clarification

The initial teacher branch is [v17 run bhqh2cyu](https://wandb.ai/maziar-sanjabi-/dressage-alfworld-gb10/runs/bhqh2cyu), `phaseswap-maxkv_q915qi2w-RANK_0`, experiment `phaseswap-maxkv-v17`, starting from the base checkpoint. Its recovered metric history covers steps 1–79. The v18 `i3lzjhcc` output log explicitly loads iteration 19, and Spark1's v17/v18 `iter_0000019` files have identical device/inode numbers with link count 2: they are the same underlying checkpoint files. For example, `common.pt` has SHA-256 `09f83e431ce037dae1113ebbd31f7072d227085c8d9dc7f7ba7bf42975ca3557` in both locations.

Therefore the relevant lineage is **v17 through 19 → v18 steps 20–94 → v18 resume 95–100**. Later v17 steps belong to a sibling branch, not the v18 continuation. This resolves the origin of the middle-checkpoint resume without guessing from directory names. It does not recover the original rollout JSONL files.

The history API returned a bounded sampled-history response with a 500-row budget and no reported truncation; all 75 and 6 expected step values in the respective recovered ranges were present. The following are arithmetic means of the returned `rollout/raw_reward` values, not held-out evaluation accuracy:

| Rollout steps | Number of step values | Mean success |
|---|---:|---:|
| 20–39 | 20 | 81.33% |
| 40–59 | 20 | 85.82% |
| 60–79 | 20 | 91.17% |
| 80–100 | 21 | 94.38% |

Steps 95–100 have respective success fractions `0.9375, 0.9140625, 0.984375, 0.96875, 0.9375, 0.984375` (mean 95.44%). These match the surviving local log. Changing sampled games/checkpoints limits interpretation of cross-step averages; these are identification/stability evidence, not a controlled learning-effect estimate.

## Original rollout availability: still unresolved

The log explicitly reports saving 128 trajectories per iteration for steps 95–100 under:

`examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories/rollout_<step>.jsonl.gz`

Those v18 files were not found in the inspected project trees on either Spark, the two `tito-mimo-alfworld-data.zip` archives, or bounded searches of Documents/Downloads/Desktop and the available dump directory. Matching filenames elsewhere belong to canonical-MIMO archives, not v18. This search does not prove there is no backup elsewhere.

The current default folders instead contain Spark1 steps 1–5 with later August 30 timestamps and Spark2 steps 40–66 with earlier August 13 timestamps. Their counts in [PLAN.md](PLAN.md) are candidate-pool audits, **not v18 dataset counts**. Neither filenames nor timestamps alone establish teacher identity.

Subsequent source matching: Spark2's mean success/reward agrees exactly with [older TITO run 22tfkup5](https://wandb.ai/maziar-sanjabi-/dressage-alfworld-gb10/runs/22tfkup5), `sync-rwfix-iter39-b`, at all 27 steps 40–66. Its saved configuration specifies TITO. This is strong attribution evidence, not a raw-token audit. The user then identified uncapped-trajectory and reward-calculation problems in this older run; **exclude it from the primary SFT corpus**, even if individual episodes look short and successful.

Both v17 and v18 logs explicitly report saving episodes to the shared default trajectories directory. The inherited sync launcher clears that directory at startup (`run_alfworld_disagg_gb10_sync_repro.sh`, lines 108–116 in the inspected checkout). This offers a concrete explanation for absent earlier files, but is not proof that no separate backup exists. Searches of readable v17/v18 W&B output logs found trajectory-write announcements, not full conversation records recoverable as the complete dataset.

Saved v18 checkpoints through `iter_0000100` remain, as does the `_hf_tito_99` export. If no original archive is recoverable, a separately approved inference job could collect fresh successful demonstrations from a frozen verified v18 checkpoint. Those would be new checkpoint-generated demonstrations, not recovered training rollouts. No such job has been launched.
