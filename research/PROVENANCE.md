# Source provenance and operational safety

Initialized 2026-10-06 as a private derivative repository with existing Git history, not a GitHub fork-network fork.

- Source: `maziars/Dressage-TITO`, branch `tito/spark-gb10`.
- Source commit: `ad3482230059035dfebdf7fc09c69e2e5265fcd6`, verified to match the inspected Spark1 checkout's HEAD and the GitHub branch.
- Slime submodule commit: `832e501fceea78ae5476be2d598b42491f639d51` (existing `.gitmodules` preserved).
- New repository: `maziars/alfworld-cue-amplification`, private, default branch `main`.
- Local source remote: `dressage-upstream`; new project remote: `origin`.
- Inherited GitHub Actions disabled at repository level before initial push.
- The original Spark1 worktree has uncommitted documentation/analysis artifacts. Those changes were not copied, committed or overwritten. Spark2's corresponding project directory has no Git metadata; runtime equivalence is not assumed.

No models, rollout datasets, credentials or machine-specific SSH keys were added as new project artifacts. No training, model inference or cluster restart was performed for initialization. CPU-only read/analysis checks do not establish GPU training correctness.

## Required before launch

1. Replace hardcoded source checkout/model/output roots with isolated project paths.
2. Remove broad process-killing and trajectory-directory cleanup from inherited launch behavior. Scope process ownership to the new run.
3. Do not reuse source checkpoint load/save directories or checkpoint-pruning/recovery scripts.
4. Mount teacher data and source weights read-only; write derived datasets, logs and student checkpoints under a new run root.
5. Verify code/runtime/dependency versions on both Sparks, with resolved configuration and checksums recorded per run.
6. Pin the same Qwen3.5 chat template and thinking mode in SFT, rollout serving and evaluation. Native local 0.8B and 4B templates differ in their default thinking mode.
7. Run format, loss-mask, grouping, cue-visibility, token-budget and checkpoint-reload tests before any expensive run.

This repository preserves the original framework; its presence here does not certify inherited commands as safe to execute.
