# Deeper timeout investigation — confirmed findings and remaining uncertainty

Read-only investigation of the historical positive-3303 evaluation on Spark2.
The recovery continues unchanged. No benchmark or backend switch was performed.

## What the logs establish

The immediate failure was the non-streaming HTTP client's 1,200-second read
deadline, not an HTTP context-length rejection. SGLang continued generating
and returning HTTP 200 responses until client cancellation propagated.

The server logged every 40 decode iterations. In the 20 minutes preceding the
first disconnect (23:54:47–00:14:47 UTC), there were 51 decode log entries,
corresponding to approximately 2,040 decode iterations, plus 944 prefill logs.
This makes a 2,048-token completion exceeding the deadline plausible, but does
**not** identify the timed-out request or prove its output length.

The first server disconnect named `29076d843b9aec150eb82c4b-turn-2`. This is not
conclusive identification: cancellation closed many requests nearly together.
On recovery this episode succeeded in eight turns; turn 2 had a 1,414-token
input and a 154-token output, and its first-turn token counts matched the old
attempt (710 input, 71 output). The interrupted original turn-2 output was not
saved. Do not describe this as proof that the timeout involved 2,048 tokens.

Among 31,991 turns belonging to the 1,927 completed original episodes:

| Quantity | Median | 90th percentile | 99th percentile | Maximum |
|---|---:|---:|---:|---:|
| Input tokens | 4,799 | 19,971 | 29,110 | 41,796 |
| Output tokens | 115 | 285 | 701 | 2,048 |

Only 18 completed turns exhausted 2,048 tokens. These statistics exclude
interrupted episodes and therefore cannot characterize the failed request.

## Substantial degradation remains unexplained by simple load counts

Twenty-minute median aggregate throughput declined from about 853 tokens/s
(21:00 window) to 258 (23:40 window), reaching approximately 192 immediately
before the failure. Decode-log waiting queues were almost always empty.

To reduce the batch-size/context confound, compare only log intervals ending
with 120–128 running requests and 1.1–1.4 million protected KV tokens:

| Metric (median) | Before 22:30 UTC | After 23:30 UTC, before failure |
|---|---:|---:|
| Qualifying intervals | 263 | 116 |
| Time per 40 decode steps, including intervening work | 8 s | 20 s |
| Running requests | 127 | 127 |
| Protected KV tokens | 1,205,813 | 1,256,838.5 |
| Intervening prefill batches | 16 | 19 |
| New prefill tokens | 10,547 | 9,118.5 |
| Cached prefill tokens reported | 275,869 | 234,459.5 |

This is an observational, approximate workload match, not a controlled benchmark.
It does not control the whole distribution of sequence lengths or kernel work.
Nevertheless, a 2.5× interval increase with similar headline workload means
"128 requests and longer contexts" is not an adequate complete diagnosis.
The preceding null-2202 run also exhibited a marked early-to-late slowdown.

The installed `_get_mamba_token_info` subtracts both available and evictable
tokens: `#full token` here is protected usage, not all cached historical tokens.
Throughput is computed from elapsed wall time between decode logs and includes
intervening prefill/CPU work; it is not isolated GPU decode-kernel throughput.

## Concrete differences from the stable RL implementation

Actual v18 RL server logs and the saved `COLOCATE_ALFWORLD_4B_GB10` preset on
Spark1 were inspected, not just the current preset defaults.

| Setting | Stable 4B RL | Current 0.8B evaluation |
|---|---|---|
| Deterministic inference | Disabled | Enabled |
| Sampling backend setting | FlashInfer | PyTorch |
| Explicit per-request sampling seeds | Not established by this inspection | Supplied on every turn |
| Triton maximum KV splits | Default 8; adaptive path | 1,024 from 262,144 / 256 |
| Mamba scheduler | `no_buffer` | `no_buffer` |
| CPU/GPU overlap | Disabled | Disabled |
| Chunked prefill | 4,096 | 4,096 |

The overlap limitation is shared with RL and is not a newly discovered
difference. RL used 128 trajectories across two nodes rather than 128 active
episodes on each node, and 30 environment steps rather than 50; workloads and
models are not directly comparable throughput benchmarks.

Installed SGLang code on Spark2 provides two concrete overhead mechanisms:

1. `layers/attention/triton_backend.py:123–139` derives deterministic maximum
   splits from full model context / fixed 256-token tile. At this model's
   262,144 context that is 1,024, not the non-deterministic default 8.
   `triton_ops/decode_attention.py:463–468` includes this maximum in the grouped
   stage-1 launch grid; `:555` loops over it in stage-2 reduction, masking
   inactive splits. Actual useful splits depend on sequence length. This is
   real overhead, **not** a claim of 128× measured runtime.
2. `layers/sampler.py:588–608` sends explicitly seeded simple sampling through
   `multinomial_with_seed`; `:545–585` hashes token positions and uses float64
   Gumbel arithmetic across the vocabulary. The model has 248,320 vocabulary
   entries: 31,784,960 entries for a batch of 128. The simple sampling path can
   bypass the configured FlashInfer/PyTorch backend distinction, so the label
   alone does not establish which kernel an RL request used. Explicit seeds
   are the relevant evaluation-side fact.

Batch-invariant matrix multiplication also uses the configured Triton path
with DeepGEMM disabled for this hardware. None of these source-level facts
alone measures their runtime share or explains the full elapsed-time trend.

## Checks and limits

During recovery, current GPU checks showed P0 state, about 2.44–2.47 GHz clocks,
75°C, and no active thermal/power slowdown flags. Available host memory was
about 28 GiB, with no swap-in/out activity in the short observation. These are
current observations, not historical telemetry at the moment of failure.
No OOM, watchdog stall, request retraction, or recompilation-limit warning was
found in the failed server log; the memory-capacity fallback warning is a
startup GB10 `nvidia-smi` reporting limitation, not an OOM.

A non-blocking `py-spy dump` of the active scheduler was denied by container
permissions. No permissions were changed and no alternate attach was attempted.
Historical logs lack per-request queue/prefill/decode timings and GPU kernel
profiles; the exact timed-out request and residual slowdown remain unresolved.

## Next diagnostic, before claiming a performance fix

Use an explicitly instrumented future run to separate queue time, prefill,
decode, sampling, and host overhead, recording request IDs and lengths plus
GPU clocks. Check whether the elapsed-time degradation persists after matching
the full active-length distribution. Profile deterministic attention/sampling
before selecting a new backend or fixed split tile size. Preserve native
262,144 context; do not silently reduce it as a speed fix.

Any backend, split-tile, sampling, or concurrency change must be versioned and
applied consistently across compared arms; it may change outputs. Do not
silently change the running endpoint evaluations. The 60-minute timeout is
recovery protection, **not a root-cause performance fix**.
