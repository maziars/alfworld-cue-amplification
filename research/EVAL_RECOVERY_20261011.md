# Positive seed 3303 evaluation recovery

User authorized investigating the failure and relaunching the remaining
evaluation on 2026-10-10 Pacific time (2026-10-11 UTC). This is recovery of
the historical **two-stage** model, not a new single-stage SFT campaign.

## Failure and evidence

Spark2's `positive-s3303-evaluation` exited with `httpx.ReadTimeout` while
awaiting response headers for a non-streaming `/generate` request. The client
allowed 1,200 seconds per read. There were 1,927 unique terminal records and
2,047 first-turn records out of the frozen 2,208-episode panel.

Immediately before the failure at approximately 00:14:47 UTC, SGLang was still
processing requests. Logged aggregate decode throughput fell from approximately
262 tokens/s at 23:50 to 199 tokens/s at 00:12:57, with about 128 requests
running. At 200 aggregate tokens/s, 128 equally progressing requests each
generating 2,048 tokens would need about 1,311 seconds, before other overhead.
This is a plausibility calculation, not a measured latency for the failed
request. The old client did not log its request ID or input length, so those
cannot be recovered conclusively from the exception.

The evidence supports a client deadline incompatible with slow, long batched
generation, not an input-context rejection or GPU OOM. The server continued to
return successful responses before mass client disconnects, then was gracefully
stopped by the pipeline's owned-process cleanup. No active GPU process remained
on Spark2 before recovery. Spark1's null-3303 evaluation was still progressing
and was left unchanged.

## Implementation

- Increase the finite non-streaming client timeout to 3,600 seconds; retain
  the 10-second connection timeout and no automatic generation retries.
- On generation exceptions, record instance ID, turn, request ID, input length,
  output budget, sampling seed, elapsed time and exception type.
- Persist successful episodes in a completion group even if another task in
  that same group raised an exception. Previously the exception could prevent
  those already completed results from being written.
- Preserve all original outputs and checkpoints. Use a new deployment copied
  from `20261010h`, with only the recovery runner and two evaluation scripts
  replaced. The separate single-stage SFT launcher edits are not deployed here.

Twenty-seven relevant local unit tests passed, including timeout configuration,
simultaneous success/error persistence, selection of missing IDs, checkpoint and
metadata validation, and rejection of duplicated or incomplete unions.

## Recovery selection and provenance

`research/recover_evaluation.py` validates the original model export and frozen
manifest, pins the original files' hashes, and selects the manifest complement
of terminal record IDs. Selection never depends on success or awareness.

- Retain all **1,927 completed episodes**, including failures.
- Evaluate **281 missing terminal episodes**, in original manifest order.
- Of these, **120 had an interrupted first turn**. Restart from turn zero with
  the original frozen seed; preserve their earlier partial attempts separately.
- Keep model, prompts, game IDs, cue assignments, seeds, temperature, 2,048-token
  generation budget, 50-turn limit, 262,144 context and 128-way concurrency.
- Continue serving profile `qwen35-08b-mamba512-concurrency128-v1`.
- After completion, create a separate 2,208-record union and report; reject
  duplicate IDs, changed metadata/checkpoint identity, or missing episodes.
  Preserve token traces in their original attempt directories.

Interrupted first turns are not additional independent trials. Canonical joint
awareness/success outcomes use the terminal attempt's first turn. A recovery
audit compares earlier and restarted first-turn outputs/awareness for all 120
interrupted episodes. Frozen seeds do not guarantee bit-identical outputs after
batch composition changes. Keep that recovery provenance in subsequent analyses.

## Locations

Container paths on Spark2:

- Original: `/workspace/tito-dressage/cue-study/runs/reduced60-association-spark2-20261010a/positive-s3303-evaluation`
- Recovery: `/workspace/tito-dressage/cue-study/runs/reduced60-positive-s3303-recovery-20261011a`
- Deployment: `/workspace/tito-dressage/cue-study/deployments/20261011a`

Recovery `status.json` tracks the wrapper, `retry/status.json` tracks the
evaluation pipeline, and `retry/eval-dev` contains new episodes. Once complete,
`combined/records.jsonl`, `combined/first_turn_records.jsonl`,
`combined/report.json` and `combined/recovery_audit.json` are authoritative for
the recovered full panel. The old campaign status deliberately remains failed;
do not edit historical status to imply the uninterrupted run succeeded.

Deployment source SHA256:

| File | SHA256 |
|---|---|
| `run_live_eval.py` | `35e76c05bf9fad9afc8e0f2f4cff6f2491833c135922014a05c6dbd4ca658c13` |
| `run_evaluation_pipeline.py` | `492a87e8e2ac526dbdabdb9e040558814640daa92fe1cf46c4075a2f1b99f0aa` |
| `recover_evaluation.py` | `6cdffc464e8a16c01d92ce59e83a748ca5551d6e79d56a3f8fbf346db7fb6295` |

Original positive-3303 checkpoint identity:
`cd8e1a0cbd5305961b75055ec49708cf5c58b716129d984b7973d832160921e8`.
