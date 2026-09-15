"""FAST batched probe. Contexts are FIXED (identical token windows for every checkpoint);
only the model weights change. Reports P(bracket tokens) at safe and fatal positions."""
import json, sys, os, glob, re, collections, pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL=sys.argv[1]; OUT=sys.argv[2]; TAG=sys.argv[3]
CTXPKL="/workspace/tito-dressage/_probe_ctx.pkl"
TOKDIR="/workspace/tito-dressage/models/Qwen3.5-4B"
tok=AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
BR={}
for i in range(len(tok)):
    s=tok.convert_ids_to_tokens(i)
    if not s: continue
    d=tok.convert_tokens_to_string([s])
    if d and set(d)=={"<"}: BR[i]=len(d)
LTN=collections.defaultdict(list)
for i,n in BR.items(): LTN[n].append(i)
def tid(s):
    e=tok.encode(s, add_special_tokens=False); return e[0] if len(e)==1 else None
WORD={w:tid(w) for w in ["env_step","function","parameter","command"]}
CTX=pickle.load(open(CTXPKL,"rb"))
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
res={}; B=4
with torch.no_grad():
    for key,ctxs in sorted(CTX.items()):
        acc=collections.defaultdict(float); N=0
        for i in range(0,len(ctxs),B):
            batch=ctxs[i:i+B]
            ii=torch.tensor(batch, device="cuda")          # all same length by construction
            try:    lg=model(ii, logits_to_keep=1).logits
            except TypeError:
                try: lg=model(ii, num_logits_to_keep=1).logits
                except TypeError: lg=model(ii).logits
            p=torch.softmax(lg[:,-1,:].float(), -1)
            for n,ids in LTN.items():
                acc[f"p_lt{n}"]+= p[:,ids].sum().item()
            acc["p_any_multi"]+= sum(p[:,ids].sum().item() for n,ids in LTN.items() if n>=2)
            for w,t in WORD.items():
                if t is not None: acc[f"p_word_{w}"]+= p[:,t].sum().item()
            N+=len(batch)
        if N: res[key]={k:v/N for k,v in acc.items()} | {"n":N}
json.dump({"tag":TAG,"model":MODEL,"res":res}, open(OUT,"w"), indent=1)
for k,v in sorted(res.items()):
    print(f"{TAG} {k:<32} n={v['n']:<4} P(<)={v.get('p_lt1',0):.5f} P(<<)={v.get('p_lt2',0):.5f} P(<<<)={v.get('p_lt3',0):.5f} P(<<<<)={v.get('p_lt4',0):.5f} P(2+)={v['p_any_multi']:.5f}")
