#!/usr/bin/env python3
# Locate where the SGLang KV + mamba pool buffers actually live on the live 4B worker.
import os, gc, socket, torch
def free_port():
    s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); return p
GIB=1024**3
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.tp_worker import TpModelWorker

def tsz(t):
    try: return t.numel()*t.element_size()
    except Exception: return 0

def dump(obj, name, depth=0, seen=None, budget=[400]):
    if seen is None: seen=set()
    if budget[0]<=0: return
    oid=id(obj)
    if oid in seen: return
    seen.add(oid); budget[0]-=1
    pad="  "*depth
    if isinstance(obj, torch.Tensor):
        if tsz(obj) > 50*1024*1024:  # >50MB
            print(f"{pad}{name}: TENSOR {tuple(obj.shape)} {obj.dtype} cuda={obj.is_cuda} param={isinstance(obj,torch.nn.Parameter)} {tsz(obj)/GIB:.2f}GiB", flush=True)
        return
    if isinstance(obj,(list,tuple)):
        for i,e in enumerate(obj[:8]): dump(e,f"{name}[{i}]",depth+1,seen,budget)
        return
    if isinstance(obj,dict):
        for k,v in list(obj.items())[:40]: dump(v,f"{name}[{k!r}]",depth+1,seen,budget)
        return
    if isinstance(obj, torch.nn.Module): return
    tn=type(obj).__name__
    if tn in ("ModelRunner","TpModelWorker","Scheduler","ServerArgs","ModelConfig"): return
    # try __dict__ and __slots__
    got=False
    d=getattr(obj,"__dict__",None)
    if isinstance(d,dict) and d:
        got=True
        for k,v in d.items(): dump(v,f"{name}.{k}",depth+1,seen,budget)
    slots=getattr(type(obj),"__slots__",None)
    if slots:
        for s in slots:
            try: dump(getattr(obj,s),f"{name}.{s}",depth+1,seen,budget); got=True
            except Exception: pass
    if depth<=2 and not got:
        refs=gc.get_referents(obj)
        print(f"{pad}{name}: <{tn}> no __dict__/slots; #referents={len(refs)}", flush=True)

MODEL="/workspace/tito-dressage/models/Qwen3.5-4B"
sa=ServerArgs(model_path=MODEL,tp_size=1,trust_remote_code=True,mem_fraction_static=0.6,
              attention_backend="triton",max_mamba_cache_size=128,context_length=49152,
              chunked_prefill_size=4096,cuda_graph_max_bs=128)
w=TpModelWorker(sa,gpu_id=0,tp_rank=0,moe_ep_rank=0,pp_rank=0,attn_cp_rank=0,moe_dp_rank=0,dp_rank=None,nccl_port=free_port())
mr=w.model_runner
print("=== token_to_kv_pool:", type(mr.token_to_kv_pool).__name__, flush=True)
dump(mr.token_to_kv_pool,"kv")
print("=== req_to_token_pool:", type(mr.req_to_token_pool).__name__, flush=True)
dump(mr.req_to_token_pool,"rtp")
print("=== allocator:", type(mr.token_to_kv_pool_allocator).__name__, flush=True)
dump(mr.token_to_kv_pool_allocator,"alloc")
# also: direct probes of the usual suspects
kv=mr.token_to_kv_pool
for attr in ["k_buffer","v_buffer","kv_buffer","data","_kv_buffer","full_kv_pool","token_to_kv_pool"]:
    v=getattr(kv,attr,None)
    if v is not None: print(f"kv.{attr}: {type(v).__name__}", flush=True)
mp=getattr(mr.req_to_token_pool,"mamba_pool",None)
print("mamba_pool:", type(mp).__name__ if mp else None, flush=True)
if mp is not None:
    mc=getattr(mp,"mamba_cache",None); print("  mamba_cache:", type(mc).__name__ if mc else None, "vars:", list(vars(mc).keys()) if mc and hasattr(mc,'__dict__') else 'NO __dict__', flush=True)
print("DONE", flush=True)
