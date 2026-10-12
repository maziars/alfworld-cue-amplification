# Runtime, code and hardware audit: strong final checkpoints

Read-only inspection of both Sparks and all six final-checkpoint evaluations.
Machine-readable evidence: `strong_end_runtime_20261011_v2.json`. Collector:
`research/audit_strong_runtime.py`. This audit launched no model inference or
training and changed no remote configurations.

## Checks that passed

- All six evaluation pipelines completed. The same saved evaluation pipeline
  SHA appears in all plans, and the same 60-game/two-cue/repeat-zero manifest
  was used (120 episodes per model).
- All 172 Python/Jinja files under deployed `dressage`/`research` match across
  nodes. The 2,279 current upstream Python/Jinja/shared-library files match
  exactly: Slime135, Megatron472, SGLang1404, ALFWorld74, TextWorld76, Triton118.
  Megatron has local modifications on both nodes, but their contents match;
  Slime Git metadata is unavailable on Spark2, so parity is established by file
  hashes rather than an assumed commit identity.
- Both nodes report NVIDIA GB10, compute capability12.1, driver580.173.02.
  Installed package versions match: Torch2.11.0a0+a6c236b9fd.nv26.3.46836102,
  SGLang0.5.9, Transformers5.3.0, Triton3.6.0+git5d72932fc5.nv26.3,
  ALFWorld0.4.2, TextWorld1.7.0, FlashInfer0.6.7+a79fb63c.nv26.3.cu132.46951218,
  Ray2.54.0. These are current audit observations, not a historical package
  installation manifest captured during training.
- Saved serving configuration matches across all six runs apart from model
  paths and small memory-pool/accounting differences. Runtime context262144,
  exclusive input limit262138, no auto-truncation, effective-concurrency profile
 128/Mamba512, deterministic inference, Triton attention and PyTorch sampling
  are common. Generation configuration, pinned template and environment render
  settings match. All boundary preflights passed.
- All **15,134 recorded generation turns** have the same sampling settings:
  max_new_tokens2048, temperature1, top_p1, top_k−1, min_p0, repetition_penalty1,
  frequency/presence penalties0, skip_special_tokens false, no_stop_trim true.
  Every recorded sampling seed follows the frozen episode-seed-plus-turn rule.
- Largest observed input is44,521 tokens, far below the effective input limit.
  No input-length mismatch can explain these final results. Some outputs reach
  the shared2048-token generation cap; their differential effects are assessed
  separately in the failure audit.

## Small measured serving differences

The auto-sized token pool ranges from6,021,813 to6,034,838 tokens (about0.22%).
Recorded memory accounting varies slightly (model1.79–1.87GB, cache68.91–69.06GB,
graphs1.62–1.75GB). Other saved server configuration fields match. These are not
evidence of an arm-specific serving setting or context cap. Deterministic mode
is configured, but no same-checkpoint cross-machine replay has been performed,
so bitwise output equivalence cannot be asserted from configuration alone.

## Hardware allocation and limits

| Seed | Null training/evaluation | Positive training/evaluation | Cued Positive−Null success |
|---|---|---|---:|
|1101|Spark1|Spark2|+8.33pp|
|2202|Spark2|Spark1|+13.33pp|
|3303|Spark1|Spark2|+6.67pp|

This is not perfect arm–machine confounding: the positive difference holds in
both assignment directions. A constant advantage of one machine cannot alone
account for that sign pattern. However, within each seed, arm is confounded
with training and serving machine. Three pairs/one reversed assignment cannot
exclude machine-by-seed effects, run-order effects, unrecorded inherited
environment variables, floating-point training variation or batch scheduling
sensitivity. Identical current code/version hashes are strong negative evidence
for code mismatch, not a complete proof of historical execution equivalence.

To isolate serving effects, evaluate the same frozen exports on the opposite
machine with identical prompts/seeds/profile and compare actual token traces.
To isolate training-hardware effects, use crossed or balanced additional
training replicas. These are recommendations, not jobs launched by this audit.
