# Token-accounting ablation: analysis code

Code and launchers for the TITO vs MIMO token-accounting study on ALFWorld with Qwen3.5-4B.

## The arms

Every arm samples identically. They differ only in **which token ids the loss is computed on**.

| arm | `token_build_mode` | extra flag | training tokens are |
|---|---|---|---|
| **TITO** (baseline) | `tito` | none | the exact sampled ids, untouched |
| **canonical MIMO** | `mimo` | none | re-rendered from the *parsed* tool call, so malformed syntax is normalised away |
| **faithful-text MIMO** | `tito` | `DRESSAGE_FAITHFUL_TEXT_RESPONSE=1` | the model's **raw text**, re-encoded; text identical to TITO, chunking may differ by BPE merges |

The faithful-text arm is the one that collapsed. See `experiments/recover_faithtext.sh`.

## Layout

- `tokenaccounting/` — offline analysis over saved rollouts. Nothing here trains.
  - Failure taxonomy: `summ.py`, `reasons.py`, `envmsg.py`, `fine.py`, `exh.py`, `dumpinv.py`, `dump.py`
  - Drift / BPE: `bpe.py`, `scan.py`, `pos.py`, `fatal.py`, `contrast.py`, `rewcond.py`, `grpo.py`
  - Parser ground truth: `ptest2.py` … `ptest5.py`, `fatal2.py` (test the **env-authoritative** SGLang `qwen3_coder` parser, not the local one)
  - Logprob probes: `mkctx5.py` builds contexts, `probe3.py` scores them, `pair.py`/`prov.py`/`turns.py` validate provenance
- `checkpoints/` — convert Megatron `torch_dist` checkpoints to HF and sweep probes over them.
  **Checkpoints are split across BOTH nodes**: rank-0 shards on the head node, rank-1 on the peer. Gather all four `.distcp` files before converting or the conversion fails on a missing shard.
- `supervisors/` — cron wedge-supervisors, one per arm.
  **WARNING:** these contain a checkpoint `prune()` whose keep-rule is `last-4 OR iter % 25 == 0`. If your save cadence never lands on a multiple of 25 the milestone branch is dead code and only the last four checkpoints survive. This destroyed iterations 4-29 of the faithtext arm. Verify the keep-set against your actual save schedule, or delete the prune call.

## Environment

Scripts take `PROJECT_ROOT`, `HOME_DIR`, `HEAD_NODE`, `PEER_NODE` from the environment. Paths and node addresses are not hardcoded.

## Data

Rollouts, drift records and training tensors are **not** in this repo. See the data release for `trajectories/` (parsed messages, gzipped JSONL, one file per iteration) and `mimo_diff/` (per-trajectory `tito_tokens` = sampled ids and `mimo_tokens` = trained ids, with per-turn diffs).

Reading `debug/*.pt`: `loss_mask` is **response-only and shorter than `tokens`**; the offset is `len(tokens) - len(loss_mask)`, which equals the prompt length. Ignoring that makes the mask appear to cover the system prompt.
