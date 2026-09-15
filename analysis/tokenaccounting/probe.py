"""Probe bracket probabilities at SAFE and FATAL positions, across checkpoints.

Positions (context ends exactly at the marked point):
  TAG_START     ...\n|<function=env_step>        -> a '<' is CORRECT here; extra brackets are the SAFE stutter
  AFTER_ONE_LT  ...\n<|function=env_step>        -> P(another '<') = the direct stutter probability
  AFTER_EQ      ...<function=|env_step>          -> ANY bracket here is FATAL (name slot corrupted)
  AFTER_SLASH   ...</|parameter>                 -> ANY bracket here is FATAL (junk swallowed into command)
"""
import json, sys, os, glob, re, collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL=sys.argv[1]; OUT=sys.argv[2]; TAG=sys.argv[3]
TOKDIR="/workspace/tito-dressage/models/Qwen3.5-4B"
tok=AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)

# --- bracket-family token ids: tokens made only of '<'
BRACKET={}
for i in range(len(tok)):
    try: s=tok.convert_ids_to_tokens(i)
    except Exception: continue
    if not s: continue
    d=tok.convert_tokens_to_string([s])
    if d and set(d)=={"<"}: BRACKET[i]=len(d)
LT1=[i for i,n in BRACKET.items() if n==1]
LTN={n:[i for i,m in BRACKET.items() if m==n] for n in sorted(set(BRACKET.values()))}
def tid(s):
    e=tok.encode(s, add_special_tokens=False); return e[0] if len(e)==1 else None
WORD={w:tid(w) for w in ["env_step","function","parameter","command","tool_call"]}
TC=tok.encode("<tool_call>", add_special_tokens=False)
print("bracket tokens:", {n:len(v) for n,v in LTN.items()}, "| <tool_call> ids:", TC, file=sys.stderr)

_pc={}
def piece(i):
    s=_pc.get(i)
    if s is None: s=tok.decode([i], skip_special_tokens=False); _pc[i]=s
    return s
def offsets(ids):
    st=[];c=0
    for i in ids: st.append(c); c+=len(piece(i))
    return st,"".join(piece(i) for i in ids)

MAXCTX=2048
def build(ids, want=40):
    st,txt=offsets(ids); b2i={p:k for k,p in enumerate(st)}
    out=collections.defaultdict(list)
    for m in re.finditer(r"<(function|parameter)=", txt):
        p=m.start()
        for nm,off in (("TAG_START",0),("AFTER_ONE_LT",1),("AFTER_EQ",m.end()-p)):
            k=b2i.get(p+off)
            if k and k>32: out[nm].append(k)
    for m in re.finditer(r"</(parameter|function|tool_call)>", txt):
        k=b2i.get(m.start()+2)
        if k and k>32: out["AFTER_SLASH"].append(k)
    res={}
    for nm,ks in out.items():
        ks=sorted(set(ks)); step=max(1,len(ks)//want)
        res[nm]=[ids[max(0,k-MAXCTX):k] for k in ks[::step][:want]]
    return res

# ---- contexts
CTX=collections.defaultdict(list)
def add_from_debug(pt, label, ntraj=8):
    d=torch.load(pt, map_location="cpu", weights_only=False)
    for s in d["samples"][:ntraj]:
        for nm,ctxs in build(list(s["tokens"]), want=12).items():
            for c in ctxs: CTX[(label,nm)].append(c)
E="/workspace/tito-dressage/experiments/faithtext_tito/debug"
add_from_debug(f"{E}/5.pt","ft_clean_r5")
add_from_debug(f"{E}/49.pt","ft_corrupt_r49")
# canonical-MIMO run: real chat template, canonical (never-corrupted) assistant text
M54="/workspace/tito-dressage/experiments/mimo54_faithful/debug"
import glob as _g
_f=sorted(_g.glob(M54+"/*.pt"))
if _f: add_from_debug(_f[0], "mimo54_canonical", ntraj=8)
M100="/workspace/tito-dressage/experiments/mimo_run_100/debug"
_f=sorted(_g.glob(M100+"/*.pt"))
if _f: add_from_debug(_f[0], "mimo100_early", ntraj=8)
print("context counts:", {f"{a}/{b}":len(v) for (a,b),v in sorted(CTX.items())}, file=sys.stderr)

# ---- score
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
res={}
with torch.no_grad():
    for (label,nm),ctxs in sorted(CTX.items()):
        acc=collections.defaultdict(float); N=0
        for c in ctxs:
            if len(c)<8: continue
            ii=torch.tensor([c], device="cuda")
            logits=model(ii).logits[0,-1].float()
            p=torch.softmax(logits, -1)
            acc["p_lt1"]+= p[LT1].sum().item()
            for k,v in LTN.items():
                if k>=2: acc[f"p_lt{k}"]+= p[v].sum().item()
            acc["p_any_multi"]+= sum(p[v].sum().item() for k,v in LTN.items() if k>=2)
            for w,i in WORD.items():
                if i is not None: acc[f"p_word_{w}"]+= p[i].item()
            acc["p_tool_call_tok"]+= p[TC[0]].item() if len(TC)==1 else 0.0
            N+=1
        if N: res[f"{label}|{nm}"]={k:v/N for k,v in acc.items()} | {"n":N}
json.dump({"tag":TAG,"model":MODEL,"res":res}, open(OUT,"w"), indent=1)
print("WROTE",OUT)
for k,v in sorted(res.items()):
    print(f"{k:<34} n={v['n']:<4} P(<)={v['p_lt1']:.4f} P(<<)={v.get('p_lt2',0):.4f} P(<<<)={v.get('p_lt3',0):.5f} P(<<<<)={v.get('p_lt4',0):.5f} P(2+)={v['p_any_multi']:.4f} P(env_step)={v.get('p_word_env_step',0):.4f} P(parameter)={v.get('p_word_parameter',0):.4f}")
