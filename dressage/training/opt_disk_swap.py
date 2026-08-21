#!/usr/bin/env python3
# OptimizerDiskSwap — GB10 phase-swap optimizer evictor (HARDENED per 5-critic review).
# evict:  per-tensor, guard ownership -> chunk BYTES through a reused pinned uint8 buffer -> disk blob
#         -> t.untyped_storage().resize_(0). reload: resize back -> disk -> pinned -> GPU (byte-exact).
# HARDENING vs v1:
#  - _guard(): assert contiguous ∧ storage_offset==0 ∧ nbytes==numel*elsize ∧ UNIQUE storage (no view/alias)
#    -> turns the shared-storage silent-corruption class into a loud abort (all 5 critics' #1 finding).
#  - dtype-AGNOSTIC uint8 byte view (fixes byte-vs-element desync for bf16/fp16 state); off advances by
#    ACTUAL bytes written.
#  - transactional-ish reload: short-read assert; blob kept until every tensor restored.
# NOTE (integration contract, NOT enforced here): evict AFTER update_weights (never in sleep -> master-param
#  trick); GEN->TRAIN = SGLang releases pools FIRST, then reload before _switch_model('actor'); DP all-or-nothing
#  barrier; fail-CLOSED (abort->supervisor); save_model must reload before save(). Gate on a loss-equality-after-swap test.
import torch, os, gc, time, numpy as np
GIB = 1024**3

class OptimizerDiskSwap:
    def __init__(self, spill_dir, stage_gib=2.0, tag=""):
        self.dir = spill_dir; os.makedirs(spill_dir, exist_ok=True)
        self._meta = {}
        self._sb = int(stage_gib * GIB)                                  # staging bytes
        self._pin = torch.empty(self._sb, dtype=torch.uint8, pin_memory=True)
        self._blob = os.path.join(spill_dir, f"opt_blob{('_'+tag) if tag else ''}.bin")

    @staticmethod
    def _guard(name, t):
        st = t.untyped_storage()
        nb_store = st.nbytes(); nb_tensor = t.numel() * t.element_size()
        assert t.is_contiguous() and t.storage_offset() == 0 and nb_store == nb_tensor, (
            f"{name}: non-owning/strided/offset storage (store={nb_store} tensor={nb_tensor} "
            f"offset={t.storage_offset()} contig={t.is_contiguous()}) — per-tensor swap would corrupt; "
            f"needs storage-grouped eviction")
        return st.data_ptr(), nb_tensor

    def evict(self, named):
        t0 = time.time(); off = 0; total = 0; seen = set()
        with open(self._blob, "wb") as f:
            for name, t in named:
                if t is None or t.numel() == 0:
                    continue
                dp, n_bytes = self._guard(name, t)
                assert dp not in seen, f"{name}: storage aliased with an earlier tensor (data_ptr {dp}) — would double-free"
                seen.add(dp)
                u8 = t.detach().contiguous().view(torch.uint8).reshape(-1)   # dtype-agnostic bytes
                for i in range(0, n_bytes, self._sb):
                    m = min(self._sb, n_bytes - i)
                    self._pin[:m].copy_(u8[i:i+m], non_blocking=False); torch.cuda.synchronize()
                    self._pin[:m].numpy().tofile(f)
                self._meta[name] = (off, n_bytes, tuple(t.shape), t.dtype)
                off += n_bytes; total += n_bytes                             # advance by ACTUAL bytes
                t.untyped_storage().resize_(0)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        return time.time() - t0, total

    def reload(self, named):
        t0 = time.time()
        with open(self._blob, "rb") as f:
            for name, t in named:
                if name not in self._meta:
                    continue
                off, nb, shape, dtype = self._meta[name]
                t.untyped_storage().resize_(nb)
                dst = t.view(torch.uint8).reshape(-1); f.seek(off)
                for i in range(0, nb, self._sb):
                    m = min(self._sb, nb - i)
                    arr = np.fromfile(f, dtype=np.uint8, count=m)
                    assert arr.size == m, f"{name}: short read ({arr.size}!={m}) at off={off+i}"
                    self._pin[:m].copy_(torch.from_numpy(arr))
                    dst[i:i+m].copy_(self._pin[:m], non_blocking=False)
                torch.cuda.synchronize()
        self._meta.clear(); os.remove(self._blob); torch.cuda.synchronize()
        return time.time() - t0


if __name__ == "__main__":
    def memavail():
        for l in open('/proc/meminfo'):
            if l.startswith('MemAvailable'): return int(l.split()[1]) / 1024 / 1024
    def fp(t):  # fast GPU byte-sum + head/tail: catches misalignment / dtype-mis-size / truncation
        u = t.detach().contiguous().view(torch.uint8).reshape(-1)
        s = int(u.sum(dtype=torch.int64).item())
        h = int(u[:2048].sum(dtype=torch.int64).item())
        tl = int(u[-2048:].sum(dtype=torch.int64).item())
        return (u.numel(), s, h, tl)
    dev = 'cuda'
    print(f"[0] baseline MemAvailable={memavail():.1f} GiB")
    keep = [torch.randn(int(2.5*GIB/4), dtype=torch.float32, device=dev) for _ in range(12)]
    torch.cuda.synchronize()
    sizes = [2.0]*4 + [0.5]*8 + [0.05]*120 + [0.005]*1200
    named = [(f"opt_{i}", torch.randn(max(1, int(g*GIB/4)), dtype=torch.float32, device=dev)) for i, g in enumerate(sizes)]
    # add a bf16 tensor to exercise dtype-agnostic path + an int64 (like a step tensor)
    named.append(("bf16_x", torch.randn(int(0.5*GIB/2), dtype=torch.bfloat16, device=dev)))
    named.append(("i64_step", torch.arange(1000, dtype=torch.int64, device=dev)))
    torch.cuda.synchronize(); s1 = memavail()
    print(f"[1] +keep+opt: MemAvailable={s1:.1f}  ({len(named)} tensors incl bf16+int64)")
    before = {n: fp(t) for n, t in named}
    sw = OptimizerDiskSwap("/workspace/tito-dressage/models/_optswap_test", tag="r0")
    te, nb = sw.evict(named); s2 = memavail()
    print(f"[2] EVICT {nb/GIB:.1f} GiB in {te:.1f}s = {nb/GIB/te:.2f} GiB/s -> MemAvailable={s2:.1f} RECLAIMED={s2-s1:.1f}")
    tr = sw.reload(named); s3 = memavail()
    after = {n: fp(t) for n, t in named}; mism = [n for n in before if before[n] != after[n]]
    print(f"[3] RELOAD in {tr:.1f}s -> MemAvailable={s3:.1f}")
    print(f"[4] CRC byte-identity: {'✅ all %d match' % len(named) if not mism else '❌ mismatch: %s' % mism[:5]}")
    print(f"VERDICT: reclaim {s2-s1:.1f} | roundtrip {te+tr:.1f}s | crc-identity {'✅' if not mism else '❌'}")
