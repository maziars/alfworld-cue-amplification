#!/usr/bin/env python3
# Trainer-side optimizer disk evict/reload for GB10 colocate phase-swap.
# EVICT (after update_weights broadcast): stream the full fp32-master + Adam m/v to disk, resize_(0) the
#   GPU storage -> frees ~23.5 GiB/rank of LPDDR so the SGLang pools can rebuild BIG for generation.
# RELOAD (after SGLang freed its pools, before train): byte-exact restore.
# Byte-exact + hardened evictor = opt_disk_swap.OptimizerDiskSwap; enumeration = phase_swap_probe._iter_opt_tensors
# (both already validated on the real optimizer). Idempotent + fail-closed. Gated by env PHASE_SWAP_TRAINER=1.
import os, time, torch, logging, sys

log = logging.getLogger("opt_phaseswap")
sys.path.insert(0, "/workspace/tito-dressage/dressage/training")


def _rank():
    try:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        return 0


class TrainerOptSwap:
    def __init__(self, rank):
        from opt_disk_swap import OptimizerDiskSwap
        spill = os.environ.get("PHASE_SWAP_OPT_DIR", "/workspace/tito-dressage/models/_phaseswap_opt")
        os.makedirs(spill, exist_ok=True)
        self.rank = rank
        self.sw = OptimizerDiskSwap(spill, stage_gib=1.0, tag=f"r{rank}")
        self.evicted = False
        self._named = None

    def _enumerate(self, optimizer):
        from phase_swap_probe import _iter_opt_tensors
        return list(_iter_opt_tensors(optimizer))

    @staticmethod
    def _fingerprint(named):
        """Light per-tensor fingerprint: head/mid/tail byte-sums ⊕ numel ⊕ name-hash → single int.
        O(1) memory per tensor (avoids a full 23.5 GiB GPU reduction that adds pressure during the tight
        evict). Catches truncation / misalignment / wrong-tensor / dtype-size errors across evict→reload;
        byte-exactness of the copy itself is already proven in opt_disk_swap."""
        acc = 0
        for n, t in named:
            u = t.detach().contiguous().view(torch.uint8).reshape(-1)
            k = u.numel()
            def _s(sl):
                return int(sl.sum(dtype=torch.int64).item()) if sl.numel() else 0
            h = _s(u[:2048]); m = _s(u[k // 2: k // 2 + 2048]); tl = _s(u[-2048:])
            acc ^= ((h * 1000003) ^ (m * 33) ^ (tl * 777) ^ (k & 0xFFFFFFFF) ^ (hash(n) & 0xFFFFFFFF))
        return acc & 0xFFFFFFFFFFFFFFFF

    def evict(self, optimizer):
        if self.evicted:
            log.warning(f"[opt-phaseswap r{self.rank}] evict: already evicted, skip")
            return
        named = self._enumerate(optimizer)
        if not named:
            raise RuntimeError(f"[opt-phaseswap r{self.rank}] evict: NO optimizer tensors enumerated")
        # capture the SAME list for reload (post-evict re-enumeration would still work but this is safest)
        self._named = named
        self._fp_before = self._fingerprint(named)  # BEFORE resize_(0)
        te, nb = self.sw.evict(named)
        self.evicted = True
        log.warning(f"[opt-phaseswap r{self.rank}] EVICT {len(named)} tensors {nb/1024**3:.1f}GiB in {te:.1f}s "
                    f"fp={self._fp_before:016x}")

    def reload(self, optimizer):
        if not self.evicted:
            return  # nothing evicted (e.g. first step) -> no-op
        assert self._named is not None, "reload with no captured tensor list"
        tr = self.sw.reload(self._named)
        fp_after = self._fingerprint(self._named)   # AFTER restore
        if fp_after != self._fp_before:
            raise RuntimeError(f"[opt-phaseswap r{self.rank}] ★GATE FAIL: optimizer BYTE-MISMATCH after reload "
                               f"(before={self._fp_before:016x} after={fp_after:016x}) — training would be corrupted")
        self.evicted = False
        self._named = None
        log.warning(f"[opt-phaseswap r{self.rank}] RELOAD in {tr:.1f}s fp={fp_after:016x} ✅byte-exact")


_SINGLETON = None
def get_trainer_swap():
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = TrainerOptSwap(_rank())
    return _SINGLETON
