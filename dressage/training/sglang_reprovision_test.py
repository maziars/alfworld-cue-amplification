#!/usr/bin/env python3
# STEP-1 mechanism test (v2): SGLang in-process pool free -> re-init (bigger mamba) -> recapture on GB10.
# v1 finding: nulling mr.* pool attrs freed ~0 (attn_backend/decode group still ref the KV pool), and the
# rebuild then hit negative rest_memory. FIX: deterministically resize_(0) the pools' backing CUDA storage
# (refcount-independent, P0-proven), null the pool + attn-backend refs, empty_cache, THEN rebuild pools +
# attention backend + graphs. Proves free actually releases LPDDR and re-init grows pools/concurrency.
import os, gc, time, socket, sys
import torch

def memavail_gib():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1024 / 1024

def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p

MODEL = "/workspace/tito-dressage/models/Qwen3.5-4B"
MAMBA_LO, MAMBA_HI = 128, 400

from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor import model_runner as MR_MOD
from sglang.srt.managers.tp_worker import TpModelWorker

_orig_imp = MR_MOD.ModelRunner.init_memory_pool
def _wrap_imp(self, total_gpu_memory, *a, **k):
    self._captured_total = total_gpu_memory
    return _orig_imp(self, total_gpu_memory, *a, **k)
MR_MOD.ModelRunner.init_memory_pool = _wrap_imp

_BLOCK_TYPES = ("ModelRunner", "TpModelWorker", "Scheduler", "ServerArgs", "ModelConfig")
def _child_objs(o):
    """Yield direct child objects of a plain instance via BOTH __dict__ and (MRO) __slots__.
    The SGLang pool classes use __slots__, which gc.get_referents / a __dict__-only walk both miss."""
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

def deep_resize0(roots):
    """resize_(0) every distinct CUDA, non-Parameter tensor reachable from `roots` (__dict__ + __slots__ +
    list/tuple/dict recursion). Returns (#freed, GiB, [top5 (GiB, shape)]). HARD guards so model weights
    are never reached: skip nn.Parameter; never descend nn.Module or runner/worker/scheduler/config;
    roots must NOT include graph_runner (back-refs mr)."""
    seen_obj, seen_store = set(), set()
    n = 0; freed_bytes = 0; sizes = []
    stack = list(roots); guard = 0
    while stack and guard < 2_000_000:
        guard += 1
        o = stack.pop()
        oid = id(o)
        if oid in seen_obj:
            continue
        seen_obj.add(oid)
        if isinstance(o, torch.nn.Parameter):
            continue  # never touch a model weight
        if isinstance(o, torch.Tensor):
            try:
                if o.is_cuda:
                    st = o.untyped_storage(); dp = st.data_ptr()
                    if dp not in seen_store and st.nbytes() > 0:
                        seen_store.add(dp); nb = st.nbytes()
                        sizes.append((nb, tuple(o.shape)))
                        st.resize_(0); n += 1; freed_bytes += nb
            except Exception:
                pass
            continue
        if isinstance(o, (list, tuple, set)):
            stack.extend(o); continue
        if isinstance(o, dict):
            stack.extend(o.values()); continue
        if isinstance(o, torch.nn.Module):
            continue  # never descend into a module (weights live here)
        if type(o).__name__ in _BLOCK_TYPES:
            continue  # defense in depth
        if type(o).__module__ == "builtins":
            continue  # int/str/dtype/etc. — no children we care about
        stack.extend(_child_objs(o))
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    top = sorted(sizes, reverse=True)[:5]
    top = [(round(b/(1024**3), 2), sh) for b, sh in top]
    return n, freed_bytes / (1024**3), top

def snap(mr, tag):
    rt = mr.req_to_token_pool
    conc = rt.size if rt is not None else None
    mamba = getattr(mr.server_args, "max_mamba_cache_size", None)
    maxtok = getattr(mr, "max_total_num_tokens", None)
    print(f"[{tag}] MemAvail={memavail_gib():.1f}GiB  concurrency(req_pool.size)={conc}  "
          f"max_total_num_tokens={maxtok}  mamba_cache_size={mamba}  "
          f"graph={'set' if getattr(mr,'graph_runner',None) is not None else 'None'}", flush=True)
    return conc, maxtok

def main():
    print(f"[boot] MemAvail={memavail_gib():.1f}GiB  building TpModelWorker (mamba={MAMBA_LO}) ...", flush=True)
    sa = ServerArgs(model_path=MODEL, tp_size=1, trust_remote_code=True,
                    mem_fraction_static=0.6, attention_backend="triton",
                    max_mamba_cache_size=MAMBA_LO, context_length=49152,
                    chunked_prefill_size=4096, cuda_graph_max_bs=128)
    t0 = time.time()
    worker = TpModelWorker(sa, gpu_id=0, tp_rank=0, moe_ep_rank=0, pp_rank=0,
                           attn_cp_rank=0, moe_dp_rank=0, dp_rank=None, nccl_port=free_port())
    mr = worker.model_runner
    print(f"[boot] worker built {time.time()-t0:.1f}s  captured_total={getattr(mr,'_captured_total',None):.2f}", flush=True)
    if getattr(mr, "graph_runner", None) is None:
        tc = time.time(); mr.init_device_graphs(); print(f"[baseline] first capture {time.time()-tc:.2f}s", flush=True)
    c0, k0 = snap(mr, "baseline")

    # ---- FREE (deterministic storage resize) ----
    # NOTE: graph_runner is EXCLUDED from the deep-walk (it back-refs mr); it's nulled below + freed by empty_cache.
    roots = [mr.token_to_kv_pool, mr.req_to_token_pool, mr.token_to_kv_pool_allocator]
    a = memavail_gib(); tf = time.time()
    nfreed, gib, top = deep_resize0([r for r in roots if r is not None])
    print(f"[free] top tensors freed (GiB, shape): {top}", flush=True)
    # now drop the object refs so init_memory_pool/init_attention_backend rebuild fresh
    mr.graph_runner = None
    mr.token_to_kv_pool_allocator = None
    mr.token_to_kv_pool = None
    mr.req_to_token_pool = None
    mr.attn_backend = None
    mr.decode_attn_backend_group = []
    mr.decode_attn_backend = None
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    te = time.time(); b = memavail_gib()
    print(f"[free] resized {nfreed} tensors (~{gib:.1f}GiB nominal) in {te-tf:.2f}s  "
          f"MemAvail {a:.1f}->{b:.1f}  RECLAIMED={b-a:.1f}GiB", flush=True)

    # ---- REBUILD bigger (mamba HI) ----
    mr.server_args.max_mamba_cache_size = MAMBA_HI
    tr = time.time(); mr.init_memory_pool(mr._captured_total); trd = time.time()-tr
    ta = time.time(); mr.init_attention_backend(); tad = time.time()-ta
    c1, k1 = snap(mr, "rebuilt")
    print(f"[rebuild] init_memory_pool {trd:.2f}s + init_attention_backend {tad:.2f}s", flush=True)

    # ---- RECAPTURE ----
    tc = time.time(); mr.init_device_graphs(); tcd = time.time()-tc
    c2, k2 = snap(mr, "recaptured")
    print(f"[recapture] init_device_graphs {tcd:.2f}s", flush=True)

    print("\n===== VERDICT =====", flush=True)
    print(f"concurrency {c0} -> {c2}  (expect {MAMBA_LO//3} -> {MAMBA_HI//3})", flush=True)
    print(f"kv max_total_num_tokens {k0} -> {k2}", flush=True)
    print(f"free reclaimed {b-a:.1f}GiB ; rebuild+recapture {trd+tad+tcd:.2f}s "
          f"(pool {trd:.2f} + attn {tad:.2f} + graph {tcd:.2f})", flush=True)
    ok = (c2 and c0 and c2 > c0 and mr.graph_runner is not None and (b-a) > 20)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  (LPDDR reclaimed + pools grew + 2nd capture ok)", flush=True)

if __name__ == "__main__":
    main()
