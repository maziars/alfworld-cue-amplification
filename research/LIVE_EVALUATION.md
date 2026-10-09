# Live seeded ALFWorld evaluation

`research/run_live_eval.py` executes a frozen manifest against an already-running
native SGLang server. It records terminal conversations in the existing scorer's
format, per-turn original input/output token IDs, finish reasons, exact requested
sampling seeds, checkpoint file hashes, generation limits, and outcome metrics.
It does not launch servers, inject awareness into live reasoning, or retry failed
episodes. Infrastructure exceptions stop the panel rather than being called task
failures. Incomplete outputs cannot be scored as a full panel.

The pinned Qwen3.5 template opens reasoning on every turn. Initial context and
new environment feedback are tokenized; generated assistant IDs are preserved
using the existing Qwen35TITOTokenizer. Missing output token IDs or a mismatch
with the server's completion count is a hard error. Local Qwen reasoning and
XML/JSON tool parsers interpret the output for action execution only. As with the
teacher's ALFWorld agent, only the first parsed tool call is executed; invalid
calls get official invalid-action feedback. All generated calls remain logged.

`sampling_params.sampling_seed` is the **actual installed SGLang 0.5.9 parameter**
(not an ignored top-level `seed`). First-turn seed is exactly the manifest's
`requested_generation_seed`; turn t uses `(seed + t) % 2**32`. Both cue conditions
in a matched pair have the same seed schedule, without implying their stochastic
paths must coincide after contexts diverge. Environments use the existing fixed
game files and deterministic reset convention, not procedurally resampled tasks.
Every native request explicitly sets top-p 1, top-k -1 (disabled), min-p 0,
repetition penalty 1, and frequency/presence penalties 0. This prevents model
generation-config defaults (e.g. top-k 20) from silently changing the evaluation
distribution. Full effective requested sampling parameters are saved per turn.

## Root-coordinated base-model server smoke on Spark2

Run only after confirming the target GPU and port are free; capture PID and log
in a new isolated run directory. Do not stop existing services. Native server
arguments are the unprefixed versions of the stable GB10 SGLang knobs:

```bash
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=0 \
python3 -m sglang.launch_server \
  --model-path /workspace/tito-dressage/models/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 30000 --tp-size 1 \
  --attention-backend triton --sampling-backend pytorch \
  --enable-deterministic-inference \
  --context-length 18432 --chunked-prefill-size 4096 \
  --max-running-requests 1 --max-mamba-cache-size 16 \
  --mem-fraction-static 0.5 --disable-cuda-graph
```

The explicit Triton attention path is the known GB10-compatible one; deterministic
mode selects PyTorch sampling. Serial evaluation keeps batch interactions out of
the initial protocol. These settings still require a live smoke on this precise
0.8B/checkpoint/runtime combination; source-level option support is not a claim
that launch has succeeded. `/health` and `/get_model_info` must be ready.

The first live server attempt exposed an additional GB10 incompatibility:
deterministic inference defaults to DeepGEMM's persistent matmul, which rejects
sm_121. The installed SGLang module explicitly supports
`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`, selecting its **batch-invariant
Triton matmul** instead. We keep deterministic inference enabled and also pin
`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=0` to forbid fallback to a
batch-variant matmul. `research/launch_eval_server.py` records both environment
settings. No upstream package edits or removal of sampling seeds are involved.

Use a **separately saved, hash-frozen four-row subset** of development IDs for
infrastructure smoke (e.g. two games, repeat zero, both cue conditions). Record
that subset and score against it, never present it as the full development panel.
Then run the unmodified `eval_dev.jsonl` for the complete panel:

```bash
python3 research/run_live_eval.py \
  --manifest /PROJECT/data/eval_dev.jsonl --manifest-sha256 SHA \
  --tokenizer /workspace/tito-dressage/models/Qwen3.5-0.8B \
  --expected-model-path /workspace/tito-dressage/models/Qwen3.5-0.8B \
  --checkpoint-label base-qwen3.5-0.8b \
  --dedicated-server \
  --output /PROJECT/runs/base_dev \
  --context 18432 --max-new-tokens 2048 --max-steps 50 --temperature 1.0
```

Keep these generation limits fixed for base, post-SFT and post-RL checkpoint
comparisons. A generation/context cap is an evaluation outcome, not permission
to truncate the SFT demonstrations. Record step-limit stops separately from token
truncation. Base capability can be evaluated on Spark2 while Spark1 trains, as
long as each evaluation is served by the recorded checkpoint and no shared
weights are modified. Checkpoint hashing requires local access to that server's
HF model files; run this client on the same machine/container as its server.

The installed SGLang rejects input plus requested output **equal to** its context
length. The runner therefore reserves one slot: at most 18,431 actual tokens
within the configured 18,432-token server context. The first live smoke exposed
this boundary and stopped with HTTP 400; its incomplete output is preserved and
not scored as a model failure. A new complete smoke run uses the corrected
budget. No SFT example is truncated by this evaluation-only rule.

The runner requires exclusive/static serving (`--dedicated-server`), snapshots
`/get_server_info`, and asserts deterministic inference, PyTorch sampling, exact
context length, and one concurrent request. It checks model info before/after
each episode; no hot checkpoint updates are allowed. Weight version alone is
not a cryptographic attestation, so dedicated serving remains an operational
requirement. Generation config/EOS settings and weights are hashed on disk.
`ALFWORLD_NAV_SPLIT=0` and `ALFWORLD_DEMANGLE=1` are asserted and recorded.
Only `stop`/`length` finishes are accepted; aborted, preempted, errored, unknown,
or missing finishes fail before an action executes. A winning action in a
length-capped response records **both** success and truncation. Environment step
limits and LLM-turn limits have distinct stop reasons.

The runner rejects held-out test panels unless `--allow-heldout-test` is given.
Development performance guides duration; test panels are reserved for the final
protocol, not routine pilot selection.

## Export the saved SFT checkpoint for generation

Use the audited wrapper `research/export_sft_checkpoint.py`, which invokes the
installed `slime/tools/convert_torch_dist_to_hf.py`. Pass the
**actual iteration directory** containing `.metadata` and `common.pt`, not the
parent `checkpoints/` directory. Use a new output path, never `--force`:

```bash
python3 research/export_sft_checkpoint.py \
  --input /PROJECT/runs/competence/checkpoints/iter_0000109 \
  --output /PROJECT/models/competence-final-hf \
  --origin /workspace/tito-dressage/models/Qwen3.5-0.8B \
  --allow-frozen-mtp --restore-fp32-linear-attn
```

Iteration numbers above are illustrative: inspect the checkpoint tracker and
save log. The HF bundle includes **153 vision and 15 MTP tensors** absent from
this text-only Megatron training path. The wrapper allows frozen vision copying;
`--allow-frozen-mtp` explicitly permits auxiliary MTP copies, only for ordinary
autoregressive serving with **speculative_algorithm=None**. Copied language-model
weights are forbidden. It validates all tensor names/shapes/dtypes, tied-embedding
handling, config/tokenizer identity, and source/output hashes, with a captured
converter log. Output and audit paths must both be new; failed attempts remain
recoverable for inspection and must not be served.

The first export caught **36 storage dtype differences**: A_log and gated norm
parameters in the 18 linear-attention layers are BF16 in Megatron but FP32 in the
origin HF bundle. `--restore-fp32-linear-attn` widens exactly those trained values
to FP32, losslessly, and records every cast. It **does not restore base values or
undo the earlier FP32→BF16 rounding** when initializing the trainer. Any other
dtype discrepancy still fails. The installed converter also overcounts
`metadata.total_size` in its HF index; the wrapper recomputes actual tensor bytes
and corrects only that informational field in its own new export, leaving
upstream code and checkpoint tensors untouched.

Therefore use the **zero-update Megatron→HF roundtrip base**, with the same export
options, as the substantive pre-SFT development baseline—not the raw HF model
used for the initial infrastructure smoke. Export the original
`models/Qwen3.5-0.8B_torch_dist/release` with the same flags plus
`--compare-origin-values`, which quantifies language-tensor numerical differences
from raw HF. This aligns the baseline with the actual trainer initialization; it
is not another cue/intervention control arm. The eval runner records exported
shard SHA256 hashes, not merely a checkpoint label.

The smoke-resume export at
`/workspace/tito-dressage/cue-study/models/smoke-resume-hf-20261009b` passed all
488-tensor audits with 36 documented widenings. Its earlier `...20261009a`
attempt intentionally failed the dtype guard and is preserved. Per-export
`export_audit.json` and sibling `.export-audit/` logs are authoritative evidence.

The zero-update baseline export
`/workspace/tito-dressage/cue-study/models/base-roundtrip-hf-20261009a` also passed.
CPU comparison covered **320 language tensors / 752,393,024 values**. Relative to
raw HF, **2,009 values in 18 linear-attention norm tensors** differ, maximum
absolute difference **0.00390625**. All other language values—including A_log
after widening—match numerically. The exported model shard SHA256 is
`8378c796cca41322d867317ef52fe843a7a79a43bdbb46971e2102f40e0cf023`.
This quantifies why the roundtrip model is the appropriate substantive baseline.
Local evidence copies: [smoke export](sft_smoke_export_verification.json) and
[base roundtrip](sft_base_roundtrip_verification.json).
