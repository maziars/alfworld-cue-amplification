# Concurrent evaluation using the Dressage rollout pattern

Implemented at the user's request; **no throughput benchmark sweep**.

## Source pattern and configuration

- `dressage/recipes/alfworld/agent_whitebox.py`: one environment and append-only
  conversation per async `rollout`, sequential reset/step operations, awaiting
  each generation call before the next action.
- `dressage/rollout/fully_async_rollout.py`: bounded active asyncio tasks,
  replenished on completion; cancellation drains tasks and closes environments.
- `dressage/proxy/proxy_client.py`: shared `httpx.AsyncClient` transport.
- Spark1's saved `compact-maxkv-v18/resolved.json`: 16 prompt groups × 8 samples
  = 128 trajectories, SGLang memory fraction 0.7, Mamba cache 128, CUDA graph
  maximum batch size 128, prefill chunk 4096, inference context 262144.

The evaluator now keeps up to **128 independent episodes active**, with one
in-flight generation per episode. It shares an async HTTP connection pool and
lets SGLang batch/queue requests. Actual admitted batch size can be lower than
128 because the runtime also limits capacity through recurrent-state pools.
The runtime-reported effective concurrency must exceed one.

Server: max-running-requests 128, memory fraction 0.7, Mamba cache 128,
CUDA graphs enabled with max batch size 128, prefill chunks 4096. Deterministic
Triton inference, explicit per-turn seeds, 262144 context, original output-token
histories, and the existing 2048-token/50-turn caps remain unchanged. Installed
SGLang's Mamba no-buffer mode still chooses the compatible overlap/cache path;
we do not force unsupported overlap scheduling or enable speculative decoding.

## Isolation and result integrity

Environment creation/reset/step/close are serialized on one dedicated worker
thread; tokenization stays on the event loop. Each episode owns its environment,
messages, token history, request IDs, and seed schedule. HTTP and environment
operations yield. No shared TextWorld environment and no concurrent mutation of
its registry. Cancellation drains in-flight worker operations before cleanup.

One writer persists outcomes and token traces immediately in completion order,
with manifest index/instance ID retained. The scorer joins by frozen IDs, not
completion order. Infrastructure errors stop the panel, cancel other tasks,
close their environments, and retain partial artifacts. There are no selective
episode retries or retries counted as new outcomes.

The serial 262k runs are preserved as partial diagnostics. The concurrent runs
start fresh on the same full 240-episode manifests; old outcomes are not mixed
with new scheduling/backend results. No SFT, association training, or RL is
launched by this change.

## Validation and runs

Tests cover bounded concurrency, exact once-only scheduling, fail-fast
cancellation, separate game state, interleaved per-episode seeds, and unchanged
TITO token prefixes, in addition to the existing tests. This is correctness
testing, not an optimization benchmark or a token-equivalence benchmark sweep.
Tests also cover HTTP transport and worker responsiveness/cancellation cleanup.
All **76 targeted tests passed**. A real ALFWorld environment also passed
create/reset/step/close on the worker thread in the Spark container.

When replacing the serial runs, 28 post-SFT outcomes (18 successes) and 11
baseline outcomes (zero successes) were preserved. Their server workers exited
on SIGTERM, but the HTTP leaders waited on stale request records beyond the
30-second cleanup timeout. After inspection confirmed the original PID,
start-time, exact argv and private group, only the old groups 18873 (Spark1) and
2568 (Spark2) were terminated with SIGKILL. No artifacts/checkpoints were deleted.
The old timeout records remain intact rather than being relabeled successful.

The first concurrent attempt (`20261010c`) passed native-context preflight on
both nodes, then failed with `httpx.RemoteProtocolError` before any episodes
completed. The likely cause was reuse of idle HTTP connections after synchronous
environment initialization exceeded the server keepalive timeout. The replacement
disables idle connection reuse (`max_keepalive_connections=0`), retains concurrent
active connections and the shared async client, and uses the Dressage 1200-second
request timeout. No generation retry was introduced. Both failed servers cleaned
up normally; their artifacts are preserved.

The `20261010d` attempt also completed native-context preflight but failed with
`httpx.ConnectTimeout` during the burst of synchronous environment startup,
before any outcomes completed. Disabling idle reuse did not prevent event-loop
starvation. Replacement `20261010e` moves environment operations to the serialized
worker above, retaining HTTP concurrency, frozen settings, and no retries.

Replacement deployment: `/workspace/tito-dressage/cue-study/deployments/20261010e`.
Replacement pipelines are tracked on 2026-10-10 UTC. Launch alone does
not establish that the panel completed; consult the live artifacts below.

| Node | Full development evaluation |
|---|---|
| Spark1 | `cue-study/runs/competence-dev-concurrent-20261010e` |
| Spark2 | `cue-study/runs/baseline-dev-concurrent-20261010e` |

`status.json` and the actual server logs are authoritative for startup and live
batching. `eval-dev/report.json` is produced only after all 240 episodes finish.

Live verification at approximately **2026-10-10 05:42 UTC**: both `20261010e`
pipelines remained running, passed native-context preflight, and reached 42
simultaneous generation requests (128 active episode slots, excess requests
queued by SGLang), with CUDA graph decoding enabled. Spark1 persisted 2 completed
episodes, both successful; Spark2 persisted 1 unsuccessful episode. These counts
only establish end-to-end operation, not accuracy: completion order favors short
episodes and neither panel is complete. The HTTP startup failures had not recurred.
