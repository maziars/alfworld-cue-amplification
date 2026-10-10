# Isolated Qwen3.5-0.8B SFT recipe

**2026-10-10 correction:** current launcher training budgets are **24,576**;
inference context is independently **262,144**. See
[the context protocol](CONTEXT_PROTOCOL_20261010.md). The 18,432 values below
describe the already-completed pilot and its historical smoke tests, not the
current defaults. That checkpoint is retained with its actual provenance.

Prepared 2026-10-09. This document describes the launch configuration, not a claim
that the GPU smoke or training completed; per-run artifacts are the evidence.

## Scope and provenance

Use the final 877-source competence dataset from
`research/data/22tfkup5_final_877_20261009`, not the earlier 878/879-source builds.
User-approved exclusions remove malformed supervision, not long trajectories.
All remaining examples fit an **18,432-token** context; the longest competence
example is **16,512** tokens. No truncation, silent filtering, or validation-game
training is allowed. Structured assistant supervision is reconstructed from saved
messages, not recovered teacher token masks.

`research/launch_sft.py` runs the installed Slime/Megatron stack inside Spark1's
existing `tito` container using a separately staged copy of this project. It
sources the actual installed `slime/scripts/models/qwen3.5-0.8B.sh`; records its
SHA, complete CLI/environment, dataset SHA, parsed training arguments, process
exit status, and finite optimizer-step metrics. It does not invoke upstream RL
launch scripts, start generation services, upgrade packages, or edit Slime.

The fresh run directory and staged code must be under a dedicated project
subdirectory visible in the existing container mount, for example
`/workspace/tito-dressage/cue-study/`. Existing checkpoints under `models/` are
read-only inputs. The launcher defaults to **dry run** and requires `--execute`
to start. It refuses an existing run directory, an existing Ray node, a busy
GPU, a mismatched dataset SHA/count, or non-training examples.

## Pilot configuration

| Setting | Value |
|---|---|
| Model | Qwen3.5-0.8B, full-parameter BF16 SFT |
| Initialization | `models/Qwen3.5-0.8B_torch_dist` on Spark1 |
| Backend | Existing Slime + Megatron, one GPU; no SGLang |
| Objective | Assistant-token NLL, per-supervised-token reduction |
| Global batch | 8 trajectories, dynamically packed microbatches |
| Context / microbatch token budget | 18,432 / 18,432 |
| Optimizer | Adam, LR 1e-5 constant, betas (0.9, 0.95), decay 0.01, clip 1.0 |
| Recompute | Full, uniform, one layer |
| Runtime memory path | Existing chunked-logprob projection, chunks 1,024 |
| Seed | 7001, for both data order and trainer |
| Steps | 110 (880 draws) |
| Saves | Every 25 rollouts and final, including optimizer/RNG |

The 877-row pool is not divisible by eight. Stock Slime's `--num-epoch 1` floors
the number of batches and would omit five rows; we instead request 110 batches.
This consumes all 877 once, then **three repeated draws** after deterministic
reshuffle. This is **880 exposures, not an exact single pass**. The saved
token/mask rollout dumps preserve source metadata so repeated IDs are auditable.
These three extra draws are computational batching, not intervention reassignment.
The competence pilot initializes from base; never initialize it from a smoke
checkpoint. Do not branch null/positive association arms until development
evaluation supports the competence checkpoint.

LR 1e-5 is a conservative initial SFT choice, not an empirically tuned optimum.
The one-pass-scale pilot limits cost before deciding whether to continue based
on development success, context exhaustion, and invalid-action rates. Never tune
duration on the held-out test set.

## Smoke protocol

Create a new one-row JSONL containing the exact longest retained competence row
with its original metadata. Record its source ID, SHA and independently audited
16,512-token length. This is a training-split smoke fixture, not a new study split.

First run, supplying actual paths and SHA:

```bash
python3 research/launch_sft.py \
  --dataset /PROJECT/data/smoke_longest.jsonl \
  --dataset-sha256 SHA --expected-rows 1 \
  --run-dir /PROJECT/runs/smoke_initial \
  --batch-size 1 --steps 1 --save-interval 1 --no-shuffle --execute
```

Then independently restart, restoring the saved optimizer and RNG as well as
model state, and train one more batch to a **new** output directory:

When `--start-rollout` is positive, the launcher adds
`--use-checkpoint-opt-param-scheduler`. Megatron then restores the checkpoint's
scheduler parameters and consumed-sample count rather than asserting that the
extended run's newly calculated decay horizon equals the original horizon.
This preserves the constant learning rate and optimizer state; it does not
restart or reset model weights. This flag is not added for fresh base starts.

```bash
python3 research/launch_sft.py \
  --dataset /PROJECT/data/smoke_longest.jsonl \
  --dataset-sha256 SHA --expected-rows 1 \
  --load /PROJECT/runs/smoke_initial/checkpoints \
  --run-dir /PROJECT/runs/smoke_resume \
  --start-rollout 1 --batch-size 1 --steps 1 \
  --save-interval 1 --no-shuffle --execute
```

Require: both exit codes zero; nonempty finite positive-loss/gradient metrics;
no OOM or truncation; expected full tokens in saved rollout dumps; checkpoint
shards plus metadata and optimizer state present; log evidence of restoration
from the new SFT checkpoint rather than base. The configured context is 18,432;
the smoke exercises the actual 16,512-token maximum, not synthetic padding to
18,432. Live ALFWorld evaluation is a separate smoke component and must also
pass before the competence pilot. Checkpoint export is required for generation.

The local Slime actor has unconditional legacy memory-history diagnostics,
including a fixed `mem_snap.pickle` path in the old repo. The project worker hook
disables those two diagnostics only inside its private workers. It does not
modify training kernels or upstream files. Chunked-logprob installation fails
closed; finite loss and gradient checks fail a run instead of allowing silent
continuation. `ray.shutdown()` only ends the Ray instance owned by this process;
there are no `pkill` or `ray stop` operations.

## Remaining execution evidence

CPU unit tests cover argument construction, tail coverage, resume arguments,
and context/mask hook selection. Full parser validation, actual model load,
forward/backward, checkpoint save/reload, and live environment evaluation must
be evidenced by the root-coordinated GPU smoke; do not infer success from the
launcher being implemented.
