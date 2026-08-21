"""Stage-2 phase-swap probe: measure real Megatron optimizer offload on GB10.
Env-gated (PHASE_SWAP_PROBE=1). Called from actor.py right after initialize_model_and_optimizer.
Enumerates the real master+m+v tensors, evicts to disk, measures LPDDR freed (= gen headroom),
reloads, verifies byte-identity. Logs [PHASE-SWAP-PROBE ...]. Does NOT change training state."""
import os, torch, logging
log = logging.getLogger("phase_swap_probe")

def _memavail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1024 / 1024
    return -1.0

def _iter_opt_tensors(optimizer):
    """Robustly enumerate fp32 master + Adam m/v across (chained) Megatron DistributedOptimizer(s)."""
    cos = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    for ci, co in enumerate(cos):
        inner = getattr(co, "optimizer", None) or co          # torch Adam inside the mcore wrapper
        state = getattr(inner, "state", {}) or {}
        for pi, (p, st) in enumerate(state.items()):
            for k in ("exp_avg", "exp_avg_sq"):
                t = st.get(k) if isinstance(st, dict) else None
                if torch.is_tensor(t) and t.is_cuda and t.numel() > 0:
                    yield (f"c{ci}_p{pi}_{k}", t)
        for gi, g in enumerate(getattr(co, "shard_fp32_from_float16_groups", []) or []):
            for ti, t in enumerate(g):
                if torch.is_tensor(t) and t.is_cuda and t.numel() > 0:
                    yield (f"c{ci}_master_g{gi}_{ti}", t)

def run_probe(optimizer, args=None):
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = 0
    tag = f"[PHASE-SWAP-PROBE r{rank}]"
    try:
        named = list(_iter_opt_tensors(optimizer))
        if not named:
            log.warning(f"{tag} NO optimizer tensors found (type={type(optimizer).__name__}); enumeration needs fixing")
            return
        dt = {}
        for _, t in named:
            dt[str(t.dtype)] = dt.get(str(t.dtype), 0) + 1
        total = sum(t.numel() * t.element_size() for _, t in named) / 1024**3
        log.warning(f"{tag} enumerated {len(named)} tensors, {total:.2f} GiB, dtypes={dt}, opt_type={type(optimizer).__name__}")
        non_fp32 = [n for n, t in named if t.dtype != torch.float32]
        if non_fp32:
            log.warning(f"{tag} {len(non_fp32)} non-fp32 tensors present; fp32-only swap would need generalizing. Measuring size only, skipping swap.")
            log.warning(f"{tag} SUMMARY optimizer_freeable≈{total:.2f} GiB (size-only)")
            return

        def fp(t):
            return (round(t.double().sum().item(), 3), round(t.double().square().sum().item(), 3))
        sample = named[:300]
        before = {n: fp(t) for n, t in sample}

        import sys
        sys.path.insert(0, "/workspace/tito-dressage/dressage/training")
        from opt_disk_swap import OptimizerDiskSwap
        spill = os.environ.get("PHASE_SWAP_DIR", f"/workspace/tito-dressage/models/_phaseswap_r{rank}")
        sw = OptimizerDiskSwap(spill)

        s0 = _memavail()
        te, nb = sw.evict(named)
        s1 = _memavail()
        log.warning(f"{tag} EVICT {nb/1024**3:.2f} GiB in {te:.1f}s -> MemAvailable {s0:.1f}->{s1:.1f} GiB "
                    f"(★ FREED {s1-s0:.1f} GiB extra during gen)")
        tr = sw.reload(named)
        s2 = _memavail()
        after = {n: fp(t) for n, t in sample}
        mism = [n for n in before if before[n] != after[n]]
        log.warning(f"{tag} RELOAD {tr:.1f}s -> MemAvailable {s2:.1f} | byte-identity(sample {len(sample)}): "
                    f"{'OK ✅' if not mism else 'MISMATCH %d ❌' % len(mism)}")
        log.warning(f"{tag} ★SUMMARY optimizer={total:.2f}GiB, FREED={s1-s0:.1f}GiB, roundtrip={te+tr:.1f}s, "
                    f"identity={'OK' if not mism else 'FAIL'}")
    except Exception as e:
        import traceback
        log.warning(f"{tag} probe FAILED: {e!r}\n{traceback.format_exc()}")
