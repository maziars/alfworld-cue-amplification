# Future-evaluation capacity update

User-authorized update, 2026-10-10. No currently running evaluation is restarted.

The original 0.8B evaluator copied the 4B RL startup cache size, but the stable
`compact-maxkv-v18` subsequently rebuilt its generation cache. Historical
Spark1 `models/_compact_run_v18.log` records cache sizes 630/661 and request
capacities 210/220. `dressage/training/sglang_phaseswap.py` also updates the
scheduler admission limit after rebuilding. Dedicated static evaluation can
instead allocate the desired cache at startup without phase-swapping.

New profile: `qwen35-08b-mamba512-concurrency128-v1`.

| Setting | Old | New |
|---|---:|---:|
| Mamba cache slots | 128 | 512 |
| Requested server concurrency | 128 | 128 |
| Effective concurrency required by preflight | 2 minimum (observed 42) | 128 minimum |
| Active episodes | 128 | 128 |
| CUDA graph maximum batch | 128 | 128 |
| Memory fraction | 0.7 | 0.7 |
| Context | 262144 | 262144 |
| Prefill chunk | 4096 | 4096 |

All task limits, manifests, checkpoints, token handling, backend, deterministic
flags and generation seeds remain unchanged. Estimated recurrent cache memory
rises from 2.35 to 9.4 GiB; attention-pool allocation adjusts within the same
memory fraction. This estimate excludes graph/workspace changes and is not a
substitute for startup validation. A new server is launched only after the
current evaluation finishes and its private server is cleaned up. The evaluator
refuses an effective request capacity below 128 and retains its full-context
boundary checks. No automatic fallback to a different experimental profile.

Deployment plan: preserve `20261010g`'s original evaluation-pipeline entry point
as a hash-checked backup, then atomically install a small forwarding entry point
for future queue invocations. The forwarder runs the new, separate `20261010h`
pipeline with a pinned script digest. Currently executing Python processes
already loaded the old code and continue untouched. Campaign plans still show
the original entry-point path; future evaluation plans record the actual new
implementation root, profile, commands and pipeline digest. Retain an explicit
routing record on each Spark rather than pretending deployment g is unmodified.

Both seed-1101 evaluations retain the old profile. Both arms of seeds 2202 and
3303 use the new profile. Treat profile as a recorded procedural difference;
do not claim all three seed pairs were run under identical serving conditions,
or interpret between-profile differences as model effects. Do not re-run seed
1101 without a separate decision.
