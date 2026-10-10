# SFT release and execution record — 2026-10-09

**Superseding context/evaluation update:** see
[CONTEXT_PROTOCOL_20261010.md](CONTEXT_PROTOCOL_20261010.md).
The new training budget is 24,576; both replacement evaluations use 262,144
inference context and actual runtime-limit checks. The older 18k runs discussed
below are historical, not the current evaluation protocol.

This page supersedes older readiness statements in the experiment, mask and
wiring documents. GPU diagnostics are not study training results.

## Training complete; development evaluation restarted

Training completed successfully at 23:29 UTC on October 9: **110 optimizer
updates, all 877 distinct sources, 880 draws** (three wraparound repeats).
The final checkpoint is `train/checkpoints/iter_0000109`; its audited HF export
is `competence-hf`, both inside the original competence run below. No training
rerun or change to the frozen data was needed.

The original post-SFT evaluation failed before any episodes on an immediate
process-identity guard. Its empty evaluation log did not preserve the observed
identity, so the exact historical mismatch cannot be recovered. A Linux startup
test of the replacement guard verified 50 real children, requiring 16 transient
argv rechecks. The guard now waits up to two seconds for the exact command while
requiring an unchanged PID/start-time and private process group. It records
mismatches, rejects persistent mismatches, and never signals an unverified group.
All 70 targeted tests passed. Original failure artifacts are preserved.

The **evaluation-only** replacement launched on Spark1 at **04:18 UTC October 10**
(21:18 PDT October 9), from isolated deployment `20261009c`:
`/workspace/tito-dressage/cue-study/runs/competence-dev-20261009b/`.
It uses the same saved model, 240 frozen development episodes, seeds, cue
conditions and inference settings. Expected model identity SHA256:
`2e84d89caf16d7af7c18831374a5e6f92295e0f5c693230b63c58ed565d11073`.
The Spark2 baseline job and deployment were not modified. No association/RL or
held-out evaluations are launched by this repair.

At the restart check, the baseline was discovered to have independently failed
at 01:31 UTC October 10 after 77 completed episodes (zero successes). The server
rejected a 18,429-token input because its input-only cap was 18,426, despite the
configured 18,432 context. This is distinct from the process-identity failure;
the shared evaluator's input-limit handling remains an open issue and may also
affect the post-SFT run. The 77 baseline outcomes are partial, not a final score.

### Original launch record

The competence pipeline was launched on **Spark1 at 23:07 UTC** on October 9,
from committed deployment `20261009b` (code commit `8d04d39`). Run directory:
`/workspace/tito-dressage/cue-study/runs/competence-877-s7001-20261009a/`.
The first optimizer update completed with loss **0.882662**, gradient norm
**15.5071**, global batch eight, and no truncation. Training is now complete;
development evaluation status is recorded separately above.

`status.json` is the pipeline state; `train/optimizer_steps.jsonl` records actual
updates. On successful training and coverage checks, the same process exports
the checkpoint, runs all 240 development episodes, and shuts down only its own
evaluation server. It stops on any failed phase. Association SFT and RL are not
queued. Training source hashes and exact configuration are saved in
`train/resolved.json`. The job runs independently of this conversation.

The matching **pre-SFT baseline development evaluation** was launched on Spark2
at 23:10 UTC, using `base-roundtrip-hf-20261009a`. Its run directory is
`/workspace/tito-dressage/cue-study/runs/baseline-dev-20261009a/`. The standalone
`run_evaluation_pipeline.py` validates the exported model hashes and all 240 dev
IDs, then manages its dedicated server and cleanup. This baseline is also an
independent job; consult its `status.json` for current progress. Both jobs avoid
all final-test panels.

## Frozen data

Current build: `research/data/22tfkup5_final_877_20261009/`.
Mask audit: `research/data/structured_sft_masks_final_877_20261009/`.

Both user-approved exclusions have been applied uniformly by source: the literal
control-token example (`316a3411…`) and the example with six unexecuted tool calls
(`7b48b9fe…`). Historical files are retained. There is no backfill, reassignment,
length-based removal, truncation, or split/evaluation manifest change.

| Dataset | Rows | Full tokens | Supervised tokens | Maximum length |
|---|---:|---:|---:|---:|
| Shared competence | 877 | 3,271,479 | 860,628 | 16,512 |
| Each association arm/seed | 1,754 | 6,550,505 | 1,722,664 | 16,519–16,520 |

There are 282 competence games. Each association file contains 877 cued examples
and 176 awareness insertions. Null has 88 insertions per cue condition; positive
has 114 cued and 62 uncued. The six association files have identical aggregate
full and supervised token counts. All 2,297 unique serializations pass the
structured-mask audit and match the inspected legacy generator on this clean
pool. Original TITO token dumps were **not recovered**: these remain explicit
parsed-message reconstructions.

Competence SHA256:
`bc0109a39062147a79320b27a2d108dea970217c08d7b0ff75cf9700d86796e7`.
All 877 retained messages and game IDs exactly match prior strict successful
environment replay inputs. This reuses verified evidence; it is not 877 fresh
replays. `prepare_sft_release.py` records input hashes and prepares the exact
longest row as a diagnostic fixture.

## Full semantic review

Two sub-agents read disjoint halves of **all original reasoning**, with code
enumeration and persisted source/turn hashes: 877 sources, 7,466 assistant turns,
2,338,305 reasoning characters. No keyword filter selected which turns to read.
Reports: [review A](semantic_review_a.json), [review B](semantic_review_b.json).
All flagged candidates were also cross-reviewed, with this additional coverage
explicitly distinguished from the main partitions. Cross-review was not blinded.
This is one LLM reader per ordinary source, not full independent double annotation
or human review, and cannot prove absence of subtle or latent awareness.

No definite being-tested/evaluation-awareness assertion was found. Ordinary
command testing and TextWorld simulation mechanics were not treated as test
detection. The previously user-adjudicated “part of a test” candidate is retained.

One new candidate is source `0c1253117d992a36d65ba42b6cd8fd4506f48019987193b862d5c3188cde549e`,
message 6: “That might reduce the chance it's scrutinized now, but I should still
hunt another.” The full trajectory concerns placing the first creditcard before
finding a second. The root and both reviewers lean retain: no evaluator, test
environment, or self-evaluation is identified. The scrutiny wording remains an
explicit ambiguity, not a confidently clean negative. The primary assistant's
release decision is **retain under the narrow being-tested criterion**; no new
exclusion is warranted by this phrase alone. Optional human adjudication has
been requested but not received; this is not labeled human-approved. The original
reviewer uncertainty is preserved in the reports.

## Recipe and execution

See [SFT_RECIPE.md](SFT_RECIPE.md) for the full configuration. The diagnostic
deployment is isolated under Spark1's
`/workspace/tito-dressage/cue-study/deployments/20261009a/`; datasets and run
outputs are in separate `cue-study/data/` and `cue-study/runs/` directories.
Existing upstream code, model inputs, services and packages are not overwritten.

The longest-row smoke (`smoke-longest-20261009a`) completed one update and saved a
checkpoint, exit code 0. It used all 16,512 tokens / 4,076 supervised tokens;
loss 1.4315135, gradient norm 14.8156, peak allocated 13.635 GB. There was no
truncation or OOM. Model/optimizer checkpoint metadata and shards are present.
The first restart attempt (`smoke-resume-20261009a`) stopped safely on a scheduler
horizon mismatch. The launcher now explicitly restores the checkpoint scheduler
for a resume. The corrected restart (`smoke-resume-20261009b`) completed a second
update and checkpoint save, exit code 0: loss 1.3460582, gradient norm 20.2996.
CPU checkpoint inspection confirms optimizer step 2 and scheduler step 2, with
102 additional changed elements in 32 small model tensors relative to the first
smoke checkpoint. This verifies real stateful continuation, not just file presence.
Live evaluation is a separate gate and has now passed: `eval-smoke-20261009b` on
Spark2 completed all four frozen development-smoke IDs, retaining exact sampled
token histories and explicit seeds. There were no missing outcomes or reasoning
blocks. The raw HF base model failed all four tasks: three context-limit stops
and one generation-length stop, with invalid-action counts recorded. These are
infrastructure-smoke observations, **not** a substantive accuracy estimate or a
cue-effect estimate. The initially failed context-boundary attempt is preserved
and excluded from model-outcome scoring. See [LIVE_EVALUATION.md](LIVE_EVALUATION.md)
for the one-token API reserve and deterministic GB10 backend configuration.

Checkpoint export passed names/shapes/dtypes/assets checks with no language-weight
substitution. Vision and unused MTP assets are retained from base; speculative
decoding is disabled. A zero-update export of the original Megatron base differs
from raw HF in 2,009 of 752,393,024 language values, all in 18 linear-attention
norm tensors. Full pre-SFT evaluation therefore uses that **roundtrip baseline**,
not the raw-HF model used for infrastructure smoke. Audits:
[smoke export](sft_smoke_export_verification.json) and
[baseline export](sft_base_roundtrip_verification.json).

66 local cue/mask/loader/revision/launcher/evaluation/export/pipeline tests pass.
Both nodes previously passed real Slime CPU-loader integration. The finalized
pipeline is `research/run_competence_pipeline.py`, default dry-run. It verifies
actual optimizer steps, checkpoint tracking, and source exposure before export
and the full 240-episode development evaluation. Only its own recorded processes
are cleaned up; failures stop the pipeline. There is no automatic association or
RL launch. The competence pilot starts from the base checkpoint, never from
diagnostic smoke weights, and uses 110 batches of eight: all 877 examples plus
three explicitly recorded wraparound draws. Development results, not held-out
test results, determine whether to extend it or branch association arms.
