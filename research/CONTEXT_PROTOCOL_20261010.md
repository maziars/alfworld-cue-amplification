# Corrected training and inference budgets — 2026-10-10

**Execution update:** the serial evaluation scheduling below is superseded by
[CONCURRENT_EVAL_20261010.md](CONCURRENT_EVAL_20261010.md), following the existing
Dressage RL concurrency pattern. Token budgets and boundary checks are unchanged.

User decision: **24,576 training tokens per GPU; 262,144 inference context**.
These are independent settings. Qwen's
[official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
specifies 262,144 native context, and the local base/exported `config.json`
has `text_config.max_position_embeddings = 262144`. No RoPE extension or model
configuration rewrite is needed.

## What has and has not changed

- Future SFT launches use 24,576 for `max_tokens_per_gpu`, `seq_length`, and
  `rollout_max_context_len`. Mask-preflight defaults and loader config agree.
- The existing competence checkpoint **was trained with 18,432**, not 24,576.
  Its 110 updates covered all 877 sources without truncation; longest input
  was 16,512. It is retained for evaluation, not relabeled or retrained.
  A different dynamic-microbatch budget can change execution/grouping, so we
  do not claim numerical equivalence to a 24,576-budget training run.
- Both replacement evaluations start the entire frozen 240-episode dev panel
  with 262,144 inference context. Old 18k partial results are diagnostic only
  and are not mixed into the new results. Model weights, data splits, seeds,
  cues, 2,048-token per-turn cap, and 50-turn limit are unchanged.
- This does not launch RL or association SFT. Future RL still needs an explicit
  policy for trajectories longer than its training budget; a large inference
  window is not authorization to truncate training trajectories silently.

## Two failures, two fixes

1. The original post-SFT launcher failed on an immediate `/proc` identity check.
   A bounded startup recheck now requires the same PID/start-time/private group
   and exact argv, logging any transient mismatches. The subsequent 18k run
   successfully generated task episodes.
2. The baseline stopped after 77 outcomes because input length 18,429 exceeded
   the serving scheduler's exclusive input threshold of 18,426. Checking only
   the configured total context missed that separate limit.

The new dedicated server entrypoint adds a **read-only** `/cue-study/limits`
endpoint. It reads the actual tokenizer-manager fields populated by the
scheduler: context length, exclusive input threshold, reserved tokens, and
auto-truncation setting. No installed SGLang files are edited.

The evaluator refuses guessed/missing limits and auto-truncation. It checks the
actual input threshold before every generation and records `context_exhausted`
locally rather than sending an invalid request. It also conservatively keeps
input plus requested output within that threshold and the reserved total
context. This leaves the serving headroom unused instead of relying on silent
scheduler output clipping. For this runtime at native context the expected
exclusive input threshold is 262,138 (largest accepted input 262,137).
If GPU capacity produces a smaller threshold, preflight stops rather than
pretending the full native context is available. HTTP errors retain their
response details and still stop the run; unrelated infrastructure failures are
never disguised as model failures.

## Verification and launch

73 targeted local tests pass, including the original failing length, both sides
of the new boundary, smaller memory-pool limits, malformed/missing limits,
disabled auto-truncation, exact token accounting, and continuation after a
locally context-exhausted episode.

Each replacement run is gated on live synthetic inference requests at **18,429**
and **262,137** input tokens, with one output token. It verifies server-reported
input/output counts, checks client rejection at/above the exclusive threshold,
and flushes the synthetic cache before study episodes. These are infrastructure
tests, not additions to the evaluation panel. Runtime evidence is saved in
`eval-dev/boundary_preflight.json`; absence of that file means the gate has not
passed. Passing it is not a guarantee against unrelated future failures.

Deployment: `cue-study/deployments/20261010b`, isolated on both Sparks.
An initial replacement attempt on Spark1 stopped before server launch because
the conservative port probe treated TCP TIME_WAIT as an occupied port. The
probe now uses SO_REUSEADDR, like the server; a regression test verifies it
still refuses a live listener. That failed run is preserved. The initial Spark2
invocation used a nonexistent manifest path and did not create a run; the
correct existing manifest was dry-run hash-validated before the replacement.

| Node | Replacement run | Model |
|---|---|---|
| Spark1 | `competence-dev-262k-20261010b` | Saved competence `iter_0000109` HF export |
| Spark2 | `baseline-dev-262k-20261010b` | Same audited zero-update roundtrip baseline |

Run paths are under `/workspace/tito-dressage/cue-study/runs/` in the `tito`
container. Jobs are detached from SSH; `status.json`, `eval.log`,
`eval-dev/records.jsonl`, and eventual `eval-dev/report.json` are the source of
truth. Original failures and the intentionally stopped old-context post-SFT
evaluation remain untouched as historical evidence.
