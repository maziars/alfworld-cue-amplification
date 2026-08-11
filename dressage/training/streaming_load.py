"""Streaming (partitioned) checkpoint LOAD as a runtime plugin — GB10 / unified-memory only.
Companion to the save-streaming fix. Opt-in via env USE_STREAMING_LOAD=1, installed through
the Ray worker_process_setup_hook (dressage.training.gb10_setup). With the env unset,
install() is a no-op.

PROBLEM (task #31): resuming an iter_N checkpoint loads model+optimizer (~14 B/param =
bf16 2 + fp32-master 4 + Adam m 4 + Adam v 4; ~58.8 GB for the 4.2B model) via one bulk
torch_dist load. On GB10 the load's transient read/reshard buffers coexist with the already-
allocated ~58.8 GB of target tensors in ONE 121.69 GiB unified pool -> peak ~117-120 GiB ->
Ray OOM at MegatronTrainRayActor.__init__ (measured 116.23/121.69 = 0.955, still climbing).
On a discrete GPU the staging buffer lives in separate host RAM, so it fits — this is a
unified-memory pathology, same class as the checkpoint-SAVE streaming fix.

FIX: bound the working set. Patch TorchDistLoadShardedStrategy.load to partition the
sharded_state_dict into K balanced groups and load each group with the ORIGINAL .load (a
normal, stock torch_dist load of ~1/K of the checkpoint, written in-place into the SAME
target tensors), freeing per-group transients between groups. Peak drops from
~(resident 58.8 + buffer 58.8) to ~(resident 58.8 + buffer 58.8/K); at K=8 that's ~66 GB.

Cause-agnostic: bounds whatever transient the load creates (reader buffering OR distributed-
optimizer reshard into fresh tensors). Correctness-preserving: identical bytes loaded into
identical tensors; the union of the K disjoint partitions == the whole state dict.

CLINICAL / GB10-only: (1) env gate USE_STREAMING_LOAD, set only in the GB10 launcher;
(2) runtime unified-memory guard (device total mem ~= system total mem => one pool);
(3) strictly correctness-preserving so it is safe even if it ever ran elsewhere.

SAFETY: install() and the patched load are fully wrapped — any failure logs and falls back
to the stock single-shot load, so this can never break a launch or a resume (worst case =
today's behavior).

NOTE: the partition/merge is validated against the live Megatron dict layout at a GPU-free
(crash) window; lists/tuples are treated as atomic units (assigned whole to one group) to
avoid index-alignment issues, which fits Megatron's dict-nested sharded_state_dict.
"""
import os

_OMIT = object()


def _enabled() -> bool:
    return os.environ.get("USE_STREAMING_LOAD", "0").lower() in ("1", "true", "yes", "on")


def _num_groups() -> int:
    try:
        return max(1, int(os.environ.get("STREAMING_LOAD_GROUPS", "8")))
    except Exception:
        return 8


def _is_unified_memory() -> bool:
    """On GB10 the CUDA device total memory ~= system total memory (one physical pool)."""
    try:
        import torch
        import psutil
        if not torch.cuda.is_available():
            return False
        dev = torch.cuda.get_device_properties(0).total_memory
        sysmem = psutil.virtual_memory().total
        return abs(dev - sysmem) / max(dev, 1) < 0.12
    except Exception:
        return False


def _atomic_units(d):
    """Yield atomic units (every non-dict node: leaf ShardedTensor/ShardedObject, or a
    list/tuple treated whole) in a stable order."""
    if isinstance(d, dict):
        for k in sorted(d.keys(), key=repr):
            yield from _atomic_units(d[k])
    else:
        yield d


def _project(d, unit_group, g):
    """Structure-preserving copy of d keeping only atomic units assigned to group g."""
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            pv = _project(v, unit_group, g)
            if pv is not _OMIT:
                out[k] = pv
        return out if out else _OMIT
    return d if unit_group.get(id(d)) == g else _OMIT


def _merge(a, b):
    """Deep dict union (lists/tuples are atomic => appear in exactly one group)."""
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def install() -> None:
    import logging
    log = logging.getLogger(__name__)
    try:
        if not _enabled():
            return
        if not _is_unified_memory() and os.environ.get("STREAMING_LOAD_FORCE", "0") != "1":
            log.warning("[streaming-load] non-unified memory; disabled (STREAMING_LOAD_FORCE=1 to override)")
            return

        from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

        if getattr(TorchDistLoadShardedStrategy.load, "_streaming_load_patched", False):
            log.warning("[streaming-load] already installed; skipping")
            return

        import gc
        import torch

        K = _num_groups()
        _orig_load = TorchDistLoadShardedStrategy.load

        def _streaming_load(self, sharded_state_dict, checkpoint_dir):
            try:
                units = list(_atomic_units(sharded_state_dict))
                if len(units) <= K:
                    return _orig_load(self, sharded_state_dict, checkpoint_dir)
                unit_group = {id(u): i % K for i, u in enumerate(units)}
                result = None
                loaded_units = 0
                for g in range(K):
                    sub = _project(sharded_state_dict, unit_group, g)
                    if sub is _OMIT or not sub:
                        continue
                    loaded = _orig_load(self, sub, checkpoint_dir)
                    loaded_units += sum(1 for _ in _atomic_units(loaded))
                    result = loaded if result is None else _merge(result, loaded)
                    del sub, loaded
                    gc.collect()
                    torch.cuda.empty_cache()
                # sanity: every unit should have been loaded exactly once
                if loaded_units != len(units):
                    log.warning(
                        f"[streaming-load] unit-count mismatch (loaded {loaded_units} != {len(units)}); "
                        "result may be incomplete — VERIFY partition logic"
                    )
                log.warning(f"[streaming-load] loaded {len(units)} units in {K} groups from {checkpoint_dir}")
                return result if result is not None else {}
            except Exception as e:
                log.warning(f"[streaming-load] partitioned load failed ({e!r}); falling back to stock load")
                return _orig_load(self, sharded_state_dict, checkpoint_dir)

        _streaming_load._streaming_load_patched = True
        TorchDistLoadShardedStrategy.load = _streaming_load
        log.warning(f"[streaming-load] runtime plugin installed (USE_STREAMING_LOAD=1, K={K})")
    except Exception as e:
        log.warning(f"[streaming-load] install failed ({e!r}); disabled (stock load unchanged)")
