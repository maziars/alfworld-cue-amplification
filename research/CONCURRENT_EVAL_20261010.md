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

Environment reset/step and tokenization stay on one event-loop thread, matching
the existing whitebox agent. Each episode owns its environment, messages,
token history, request IDs, and seed schedule. Only HTTP generation yields.
No shared TextWorld environment and no concurrent mutation of its registry.

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

Deployment: `/workspace/tito-dressage/cue-study/deployments/20261010c`.

| Node | Full development evaluation |
|---|---|
| Spark1 | `cue-study/runs/competence-dev-concurrent-20261010c` |
| Spark2 | `cue-study/runs/baseline-dev-concurrent-20261010c` |

`status.json` and the actual server logs are authoritative for startup and live
batching. `eval-dev/report.json` is produced only after all 240 episodes finish.
