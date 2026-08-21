#!/usr/bin/env python3
# GB10 colocate phase-swap — REAL free/rebuild of SGLang KV+mamba pools in-process, repurposing the
# release/resume_memory_occupation control path (which slime's colocate flow already calls, but which is a
# NOOP on GB10 because torch_memory_saver is broken). Gated by env SGLANG_PHASESWAP=1.
#
#   release_memory_occupation  -> FREE KV+mamba+cuda-graph pools (KEEP weights resident) -> returns ~61 GiB LPDDR
#   resume_memory_occupation   -> REBUILD pools at SGLANG_GEN_MAMBA_CACHE_SIZE (default 400), re-profiling the
#                                 now-larger free memory (get_available_gpu_memory=psutil on integrated GB10),
#                                 re-init attention backend + cuda graphs, and RE-WIRE the scheduler refs.
#
# Injected by a 2-line edit at the top of run_scheduler_process (runs in the scheduler subprocess, before
# Scheduler/ModelRunner construction, so the init_memory_pool total-capture is in place at startup).
# All logic lives here (our repo) so it is portable + iterable without re-patching dist-packages.
import os, gc, time, torch, logging

logger = logging.getLogger("phaseswap")
GIB = 1024**3
_BLOCK_TYPES = ("ModelRunner", "TpModelWorker", "Scheduler", "ServerArgs", "ModelConfig")


def _enabled():
    return os.environ.get("SGLANG_PHASESWAP") == "1"


def _memavail():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"):
                return int(l.split()[1]) / 1024 / 1024
    except Exception:
        return -1.0


# ---------- deterministic pool free (resize_(0) backing storage; __slots__-aware; weight-safe) ----------
def _child_objs(o):
    d = getattr(o, "__dict__", None)
    if isinstance(d, dict):
        yield from d.values()
    for klass in type(o).__mro__:
        slots = getattr(klass, "__slots__", None)
        if not slots:
            continue
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots:
            try:
                yield getattr(o, s)
            except Exception:
                pass


def _deep_resize0(roots):
    seen_obj, seen_store = set(), set()
    n = 0; freed = 0
    stack = list(roots); guard = 0
    while stack and guard < 2_000_000:
        guard += 1
        o = stack.pop()
        i = id(o)
        if i in seen_obj:
            continue
        seen_obj.add(i)
        if isinstance(o, torch.nn.Parameter):
            continue
        if isinstance(o, torch.Tensor):
            try:
                if o.is_cuda:
                    st = o.untyped_storage(); dp = st.data_ptr()
                    if dp not in seen_store and st.nbytes() > 0:
                        seen_store.add(dp); freed += st.nbytes(); st.resize_(0); n += 1
            except Exception:
                pass
            continue
        if isinstance(o, (list, tuple, set)):
            stack.extend(o); continue
        if isinstance(o, dict):
            stack.extend(o.values()); continue
        if isinstance(o, torch.nn.Module):
            continue
        if type(o).__name__ in _BLOCK_TYPES:
            continue
        if type(o).__module__ == "builtins":
            continue
        stack.extend(_child_objs(o))
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    return n, freed / GIB


def _mr(self):
    return self.tp_worker.model_runner


def _ck(tag):
    try:
        torch.cuda.synchronize()
        logger.warning(f"[phaseswap] free.ck {tag} OK")
    except Exception as e:
        logger.warning(f"[phaseswap] free.ck {tag} CUDA-ERR: {type(e).__name__}: {str(e)[:100]}")
        raise


def _phaseswap_free(self):
    """Free KV+mamba+cuda-graph pools (KEEP model weights). Idempotent: no-op if already freed."""
    mr = _mr(self)
    if getattr(mr, "token_to_kv_pool", None) is None and getattr(mr, "req_to_token_pool", None) is None:
        logger.info("[phaseswap] free: pools already freed, skip")
        return 0.0
    t0 = time.time(); a = _memavail()
    _ck("entry")
    # ★ GB10-colocate CUDA-graph rule: DO NOT destroy the decode graph here. empty_cache() with NO live graph
    # (i.e. right after tearing one down) throws cudaErrorInvalidValue in the shared-GPU colocate context.
    # We resize the KV/mamba backing STORAGE (deep_resize0's empty_cache runs with the graph STILL ALIVE ->
    # frees the ~24 GiB to the OS), and LEAVE graph_runner intact. The stale graph now points at empty buffers
    # but is never replayed (engine paused); init_device_graphs() at rebuild replaces it, recapturing a NEW
    # graph BEFORE its own empty_cache, so a live graph always exists whenever empty_cache runs.
    roots = [x for x in (mr.token_to_kv_pool, mr.req_to_token_pool, mr.token_to_kv_pool_allocator) if x is not None]
    n, gib = _deep_resize0(roots)
    _ck("after-resize")
    mr.token_to_kv_pool_allocator = None
    mr.token_to_kv_pool = None
    mr.req_to_token_pool = None
    mr.attn_backend = None
    try:
        mr.decode_attn_backend_group = []
        mr.decode_attn_backend = None
    except Exception:
        pass
    # drop scheduler-side refs too (rebuilt on resume). graph_runner is intentionally LEFT ALIVE.
    self.req_to_token_pool = None
    self.token_to_kv_pool_allocator = None
    self.tree_cache = None
    b = _memavail()
    logger.info(f"[phaseswap] FREE {n} tensors ~{gib:.1f}GiB in {time.time()-t0:.2f}s  MemAvail {a:.1f}->{b:.1f} (+{b-a:.1f})")
    return gib


def _build_tree_cache(self):
    """Rebuild self.tree_cache from current pools — mirrors Scheduler.__init__ (scheduler.py:657-727)."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    sa = self.server_args
    params = CacheInitParams(
        disable=sa.disable_radix_cache,
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        page_size=self.page_size,
        is_eagle=self.spec_algorithm.is_eagle(),
        tp_cache_group=(self.attn_tp_cpu_group if sa.enable_dp_attention else self.tp_cpu_group),
        eviction_policy=sa.radix_eviction_policy,
        # enable_metrics=False on REBUILD: the startup tree_cache already registered its Prometheus collector
        # in the global registry; re-registering the same timeseries on rebuild -> ValueError (duplicated
        # timeseries) kills the scheduler. Metrics are monitoring-only, not correctness. (Only bites when the
        # run has --enable-metrics, i.e. colocate; the standalone repro has metrics off so it never saw this.)
        enable_metrics=False,
        enable_kv_cache_events=self.enable_kv_cache_events,
        enable_mamba_extra_buffer=sa.enable_mamba_extra_buffer(),
        pp_rank=self.pp_rank,
        pp_size=self.pp_size,
        chunked_prefill_size=sa.chunked_prefill_size,
        sliding_window_size=self.sliding_window_size,
    )
    if sa.chunked_prefill_size is not None and sa.disable_radix_cache:
        if not self.is_hybrid_swa:
            from sglang.srt.mem_cache.chunk_cache import ChunkCache
            self.tree_cache = ChunkCache(params)
        else:
            from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
            self.tree_cache = SWAChunkCache(params)
    elif self.enable_hierarchical_cache:
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        self.tree_cache = HiRadixCache(params=params, server_args=sa)
        self.tp_worker.register_hicache_layer_transfer_counter(self.tree_cache.cache_controller.layer_done_counter)
    elif self.is_hybrid_swa:
        from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
        self.tree_cache = SWARadixCache(params=params)
    elif self.is_hybrid_ssm:
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
        self.tree_cache = MambaRadixCache(params)
    else:
        from sglang.srt.mem_cache.radix_cache import RadixCache
        self.tree_cache = RadixCache(params)


def _phaseswap_rebuild(self, mamba):
    """Rebuild KV+mamba pools, re-init attn + cuda graphs, re-wire scheduler. Idempotent.
    `mamba`: int -> force that max_mamba_cache_size; None -> AUTO-size from current free mem (biggest that
    fits given the freed optimizer) via SGLang's mamba_full_memory_ratio path — matches 'as large as fits'."""
    mr = _mr(self)
    if getattr(mr, "token_to_kv_pool", None) is not None:
        logger.info("[phaseswap] rebuild: pools already present, skip")
        return
    assert hasattr(mr, "_captured_total"), "phaseswap: init_memory_pool total not captured (install() ran?)"
    t0 = time.time(); a = _memavail()
    mr.server_args.max_mamba_cache_size = None if mamba is None else int(mamba)
    mr.init_memory_pool(mr._captured_total)
    mr.init_attention_backend()
    mr.init_device_graphs()
    # re-wire scheduler refs
    self.req_to_token_pool, self.token_to_kv_pool_allocator = self.tp_worker.get_memory_pool()
    _build_tree_cache(self)
    self.init_schedule_policy()
    # recompute max_running_requests (mirror ModelRunner.initialize) across mr / tp_worker / scheduler
    sa = mr.server_args
    mrr = (mr.max_total_num_tokens // 2 if sa.max_running_requests is None
           else sa.max_running_requests // mr.dp_size)
    mrr = min(mrr, mr.req_to_token_pool.size)
    mr.max_running_requests = mrr
    self.tp_worker.max_running_requests = mrr
    self.max_running_requests = mrr
    # ★ THE ADMISSION CAP: get_next_batch admits up to get_global_server_args().pp_max_micro_batch_size
    # (scheduler.py:1955), which was frozen at init from the STARTUP max_running_requests (42). Without
    # updating it here, growing mamba/req_to_token_pool.size does nothing — #running-req stays 42. Update it.
    try:
        from sglang.srt.server_args import get_global_server_args
        get_global_server_args().pp_max_micro_batch_size = max(mrr // max(self.pp_size, 1), 1)
    except Exception as e:
        logger.warning(f"[phaseswap] pp_max_micro_batch_size update failed: {e}")
    b = _memavail()
    logger.info(f"[phaseswap] REBUILD mamba={mamba} concurrency={mr.req_to_token_pool.size} "
                f"max_running_requests={mrr} max_total_num_tokens={mr.max_total_num_tokens} "
                f"in {time.time()-t0:.2f}s  MemAvail {a:.1f}->{b:.1f}")


# ---------- release/resume overrides (reuse existing plumbing + slime call sites) ----------
def _has_kv_tag(tags):
    # KV-cache tag present, or empty/None (=all types). slime: offload->release() [None];
    # onload_weights->resume([WEIGHTS]); onload_kv->resume([KV_CACHE, CUDA_GRAPH]).
    if not tags:
        return True
    try:
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
        return GPU_MEMORY_TYPE_KV_CACHE in tags
    except Exception:
        return any("kv" in str(t).lower() for t in tags)


def _make_release(orig):
    def release_memory_occupation(self, recv_req):
        if not _enabled():
            return orig(self, recv_req)
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqOutput
        tags = recv_req.tags or []
        for tag in tags:
            self.offload_tags.add(tag)
        if not _has_kv_tag(recv_req.tags):
            # weights-only release: nothing to do (we KEEP weights resident for re-broadcast).
            return ReleaseMemoryOccupationReqOutput()
        assert self._is_no_request(), "release_memory_occupation called with ongoing requests"
        # PAUSE the batch loop so its idle self-check + get_next_batch stop touching the (freed) pools.
        # (event_loop_normal: process_input_requests runs BEFORE `if self._engine_paused: continue`, so
        #  control msgs like resume are still received while paused.)
        self._engine_paused = True
        _phaseswap_free(self)
        torch.get_device_module().synchronize()
        return ReleaseMemoryOccupationReqOutput()
    return release_memory_occupation


def _make_resume(orig):
    def resume_memory_occupation(self, recv_req):
        if not _enabled():
            return orig(self, recv_req)
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqOutput
        tags = recv_req.tags or []
        for tag in tags:
            self.offload_tags.discard(tag)
        if not _has_kv_tag(recv_req.tags):
            # weights-only resume (slime onload_weights): weights were never freed -> no-op. Stay paused
            # until onload_kv rebuilds the pools (which happens AFTER the trainer evicts its optimizer).
            return ResumeMemoryOccupationReqOutput()
        _v = os.environ.get("SGLANG_GEN_MAMBA_CACHE_SIZE", "auto")
        mamba = None if _v.lower() == "auto" else int(_v)
        _phaseswap_rebuild(self, mamba)
        # UNPAUSE: pools + tree_cache are valid again, safe for the idle self-check / get_next_batch.
        self._engine_paused = False
        return ResumeMemoryOccupationReqOutput()
    return resume_memory_occupation


_INSTALLED = False
def install():
    """Monkeypatch ModelRunner.init_memory_pool (stash total) + Scheduler release/resume. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.scheduler import Scheduler

    _orig_imp = ModelRunner.init_memory_pool
    def _wrap_imp(self, total_gpu_memory, *a, **k):
        self._captured_total = total_gpu_memory
        return _orig_imp(self, total_gpu_memory, *a, **k)
    if not getattr(ModelRunner.init_memory_pool, "_phaseswap", False):
        _wrap_imp._phaseswap = True
        ModelRunner.init_memory_pool = _wrap_imp

    Scheduler._phaseswap_free = _phaseswap_free
    Scheduler._phaseswap_rebuild = _phaseswap_rebuild
    Scheduler._phaseswap_build_tree_cache = _build_tree_cache
    Scheduler.release_memory_occupation = _make_release(Scheduler.release_memory_occupation)
    Scheduler.resume_memory_occupation = _make_resume(Scheduler.resume_memory_occupation)

    # flush_cache no-op while pools are freed: slime's update_weights() calls engine.flush_cache() during the
    # free->rebuild window, but stock flush_cache does self.tree_cache.reset()/req_to_token_pool.clear() which
    # we've nulled -> AttributeError kills the scheduler. Nothing to flush when the KV cache is gone.
    _orig_flush = Scheduler.flush_cache
    def _flush_cache(self):
        if not _enabled():
            return _orig_flush(self)
        if getattr(self, "tree_cache", None) is None or getattr(_mr(self), "token_to_kv_pool", None) is None:
            logger.info("[phaseswap] flush_cache: pools freed, no-op success")
            return True
        return _orig_flush(self)
    Scheduler.flush_cache = _flush_cache

    # self_check_during_idle no-op while pools are freed: slime's update_weights() calls continue_generation()
    # which UNPAUSES the engine mid free->rebuild window; the idle self-check then does
    # tree_cache.supports_mamba() on our nulled tree_cache -> AttributeError kills the scheduler. Skip the
    # self-check when there's nothing to check (pools gone); it runs normally once the rebuild restores them.
    _orig_selfcheck = getattr(Scheduler, "self_check_during_idle", None)
    if _orig_selfcheck is not None:
        def _self_check_during_idle(self):
            if _enabled() and (getattr(self, "tree_cache", None) is None
                               or getattr(_mr(self), "token_to_kv_pool", None) is None):
                return
            return _orig_selfcheck(self)
        Scheduler.self_check_during_idle = _self_check_during_idle

    _INSTALLED = True
    logger.info(f"[phaseswap] installed (enabled={_enabled()}, gen_mamba={os.environ.get('SGLANG_GEN_MAMBA_CACHE_SIZE','400')})")
