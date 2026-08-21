# GB10 colocate phase-swap — deploy patches

These scripts apply the GB10 colocate phase-swap that lives **outside** the repo-tracked
source (installed `dist-packages/sglang`), plus the anchored edits to the slime submodule.
They are idempotent + reversible and must be **re-applied per node after a container rebuild**
(they survive `docker restart`, not `docker rm`/recreate). Deploy to BOTH nodes.

Pipe each to the container Python: `docker exec -i tito python3 - < <script>.py`

## sglang dist-packages injects (NOT in any repo — the only unreproducible part)
- `patch_sched_inject.py` — 2-line inject at the top of `run_scheduler_process` in
  `sglang/srt/managers/scheduler.py`: `sys.path.insert(/workspace/tito-dressage)` +
  `dressage.training.sglang_phaseswap.install()`. Dormant unless `SGLANG_PHASESWAP=1`.
- `patch_sched_disable_graph.py` — sets `server_args.disable_cuda_graph=True` when
  `SGLANG_PHASESWAP=1` (workaround for the empty_cache-after-graph-destroy trap; benign for
  this bandwidth-bound 4B — cuda-graph is a measured no-op on GB10).
- `patch_sched_disable_memsaver.py` — sets `server_args.enable_memory_saver=False` when
  `SGLANG_PHASESWAP=1`. **THE KEY FIX**: with memory_saver on, `resize_(0)` frees the KV
  tensor but the torch_memory_saver allocator retains the LPDDR pages (gen->train free
  returns +0), so training OOMs; off => the free returns ~20-25 GiB to the OS.
- `patch_graph_toggle.py` — makes the disable-graph inject env-toggleable via
  `SGLANG_PS_KEEP_GRAPH=1` (default still disables). cuda-graph proven compatible with the
  free/rebuild once memory_saver is off, but a measured no-op here — keep OFF.

## slime submodule edits (now committed to the slime `gb10-patches` branch; kept here as deploy docs)
- `patch_actor_phaseswap.py`, `patch_actorgroup_phaseswap.py`, `patch_train_phaseswap.py`,
  `patch_placement_skip_offload.py` — the actor/fan-out/train-loop/placement wiring.

## dressage run_config edit (committed; deploy doc)
- `patch_preset_offload.py` — emit `--offload-rollout` when `PHASE_SWAP_TRAINER=1`.

## Launch env (set at ray-start AND the launcher docker-exec, both nodes)
`SGLANG_PHASESWAP=1 PHASE_SWAP_TRAINER=1 SGLANG_GEN_MAMBA_CACHE_SIZE=auto PHASE_SWAP_OPT_DIR=/workspace/tito-dressage/models/_phaseswap_opt`
Also required: `--set use_streaming_load=False` (2-node streaming-load desync), `--set sglang_mem_fraction=0.4`.
Optional ALFWorld efficiency: `ALFWORLD_NAV_SPLIT=1` (drops the constant nav list; prompt-changing — ablation only).
