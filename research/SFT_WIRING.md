# Structured SFT loader wiring

**2026-10-10:** current SFT training/loader budgets are **24,576**, independent
of the **262,144** inference context. See
[CONTEXT_PROTOCOL_20261010.md](CONTEXT_PROTOCOL_20261010.md).
Any 18,432-token wiring examples below describe the earlier pilot.

**Current status:** see [SFT_RELEASE_20261009.md](SFT_RELEASE_20261009.md). The
877-source build is deployed in isolation and GPU smoke execution has begun.
The following sections preserve the earlier CPU integration record.

2026-10-06. **Project-local integration implemented and CPU-tested on both Sparks.
No training launch or persistent remote deployment.**

**2026-10-09 update:** the embedded-control-token source has now been excluded as
authorized, in a new 878-source draft. The separate replay exception is resolved
under historical execution semantics, with ignored-call and semantic-review
caveats remaining. See [DATA_REVIEW_20261009.md](DATA_REVIEW_20261009.md) for current
paths and results; the blocked-source discussion below describes the earlier build.

## Runtime entry point

```text
dressage.training.structured_sft.generate_rollout
```

The hook replaces the stock `slime.rollout.sft_rollout.generate_rollout` selection,
not Slime's source code. The mask implementation now lives in the installed
`dressage.training.sft_masks` package. `research/sft_masks.py` remains a compatibility
import for existing audits. The fixed Jinja template is included as package data.

Use the original **per-arm message JSONL** from the dataset builder, retaining
all paired/duplicate rows. The loader reads `prompt` and `metadata.tools`, renders
with the pinned thinking-preserving template, and assigns:

- `sample.tokens`: full input IDs.
- `sample.response_length`: tokens from the first supervised position onward.
- `sample.loss_mask`: the corresponding response-only mask, including zeros on
  intervening environment observations.
- `sample.reward = 0`: a placeholder for the SFT backend, not a task/awareness reward.

It returns Slime's expected nested singleton sample groups. It does not inject
anything, reassign interventions, shift targets, or call an environment/model.

## Configuration fragment

[sft_loader_config.json](sft_loader_config.json) records the namespace values and
CLI fragment. It is **not a complete training recipe or an automatically loaded
Slime configuration file**. The eventual isolated launcher must supply:

```text
--rollout-function-path dressage.training.structured_sft.generate_rollout
--input-key prompt
--metadata-key metadata
--n-samples-per-prompt 1
--loss-type sft_loss
--calculate-per-token-loss
--disable-compute-advantages-and-returns
--debug-train-only
--rollout-max-context-len 18432
--seq-length 18432
--rollout-shuffle
```

Keep the default global dataset enabled. **Do not** pass `--apply-chat-template`
or `--disable-rollout-global-dataset`. The CLI flag disabling advantages maps to
the namespace field `compute_advantages_and_returns=False`, not a `disable_*`
field. The integration test uses Slime's real parser to check these settings.

Still supply the dataset and Qwen3.5-0.8B checkpoint paths, architecture arguments,
batch sizes, per-GPU token budget, optimizer/epoch choices, save directory and
hardware settings in the eventual launch recipe. This fragment does not establish
GPU feasibility at 18,432 tokens. Do not reuse the upstream example's broad
process-killing commands or its Qwen3-4B / eight-GPU settings.

## Guards

The loader rejects wrong objective/advantage settings, multiple samples per
prompt, pre-rendered string prompts, non-training split metadata, missing
provenance/tools, unnormalized tool arguments, tokenizer/template drift, and
over-context batches. It never silently truncates, filters or repairs examples.

Before fetching a batch, it checks the **entire source dataset's message
structure**. The unresolved literal control-token example therefore fails before
the first batch is fetched, not after earlier gradient updates. All 11,427 draft
rows were checked: the only structural blocker is the previously identified
source, appearing in 13 rows across the seven datasets. It remains unchanged.
Token-length/mask checks apply to the entire fetched batch before any of that
batch's training fields are mutated. The separate full-corpus mask audit remains
necessary before launch.

## Verification

- **37 local tests pass:** 21 cue-study tests, 9 mask tests, 7 loader tests.
- Both Spark containers passed a CPU-only integration test with actual Slime
  `Dataset`, `RolloutDataSourceWithBuffer`, `Sample`, and CLI option parsing.
- Four fixture rows per node: a normal trajectory, the longest clean competence
  trajectory (16,512 tokens), a cue+awareness example, and a deliberate duplicate.
- Full token arrays and response-only masks match the independent stored audit
  exports exactly. Prompts, metadata, row multiplicities and ordering are preserved.
- Adding the unchanged problematic source to the temporary fixture pool blocks
  loading before the buffer offset advances.
- CUDA remained uninitialized. No weights, Ray cluster, training job, or service
  was started. Temporary code/data fixtures were isolated and removed afterward;
  neither existing Spark checkout was modified.

Successful reports: `research/data/sft_loader_smoke_spark1_v3/report.json` and
`research/data/sft_loader_smoke_spark2_v3/report.json`. v1 tested the sample path
before the parser audit; v2 stopped because the standalone parser test omitted
its required fixture batch size. v3 is the completed test of the final wiring.

Both nodes reported Transformers 5.3.0, tokenizers 0.22.2, Ray 2.54.0, and Torch
`2.11.0a0+a6c236b9fd.nv26.3.46836102`. No packages were upgraded.

Reproduce with a new output directory:

```bash
.venv/bin/python research/smoke_sft_loader.py \
  --ssh spark1 \
  --build research/data/22tfkup5_length_audit_v2/draft_datasets \
  --masks research/data/structured_sft_masks_v1 \
  --output research/data/sft_loader_smoke_NEW
```

## Before launch

Resolve the embedded-control-token data-quality decision, complete the remaining
corpus/evaluation release checks, persistently deploy the isolated project and
launcher, and separately authorize a GPU forward/backward/checkpoint smoke test.
The loader wiring is complete; those are not claimed complete by this CPU test.
