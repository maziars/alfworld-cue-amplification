"""Build ONE fixed context set, saved to disk, reused unchanged for every checkpoint.
Context sets are kept SEPARATE by trajectory source: TITO run, faithful-text, canonical MIMO, MIMO-100."""
import json, glob, gzip, os, re, collections, pickle, sys
import torch
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
W=1024          # fixed window: every context is exactly W tokens
PER=48          # contexts per (source, position)
_pc={}
def piece(i):
    s=_pc.get(i)
    if s is None: s=TOK.decode([i], skip_special_tokens=False); _pc[i]=s
    return s
def offsets(ids):
    st=[];c=0
    for i in ids: st.append(c); c+=len(piece(i))
    return st,"".join(piece(i) for i in ids)
def find(ids):
    st,txt=offsets(ids); b2i={p:k for k,p in enumerate(st)}
    out=collections.defaultdict(list)
    for m in re.finditer(r"<(function|parameter)=", txt):
        p=m.start()
        for nm,off in (("TAG_START",0),("AFTER_ONE_LT",1),("AFTER_EQ",m.end()-p)):
            k=b2i.get(p+off)
            if k and k>=W: out[nm].append(k)
    for m in re.finditer(r"</(parameter|function|tool_call)>", txt):
        k=b2i.get(m.start()+2)
        if k and k>=W: out["AFTER_SLASH"].append(k)
    return out
CTX=collections.defaultdict(list)
def harvest(ids, src):
    for nm,ks in find(ids).items():
        for k in sorted(set(ks)):
            if len(CTX[(src,nm)])<PER: CTX[(src,nm)].append(ids[k-W:k])
def from_debug(pt, src, ntraj=12):
    try: d=torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as e: print("skip",pt,e, file=sys.stderr); return
    for s in d["samples"][:ntraj]: harvest(list(s["tokens"]), src)
from_debug("/workspace/tito-dressage/experiments/faithtext_tito/debug/5.pt","faithtext_r5_clean")
from_debug("/workspace/tito-dressage/experiments/faithtext_tito/debug/49.pt","faithtext_r49_corrupt")
f=sorted(glob.glob("/workspace/tito-dressage/experiments/mimo54_faithful/debug/*.pt"))
if f: from_debug(f[0],"canonicalMIMO_traj")
f=sorted(glob.glob("/workspace/tito-dressage/experiments/mimo_run_100/debug/*.pt"))
if f: from_debug(f[0],"mimo100_traj")
# TITO-run trajectories rendered through the REAL chat template
sys.path.insert(0,"/workspace/tito-dressage")
from dressage.recipes.alfworld.tools import ALFWORLD_TOOL_SCHEMAS
TJ="/workspace/tito-dressage/examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories/rollout_1.jsonl.gz"
n=0
for line in gzip.open(TJ,'rt'):
    t=json.loads(line); msgs=t.get("messages") or []
    if len(msgs)<10: continue
    conv=[]
    for m in msgs:
        r=m.get("role")
        if r in ("system","user","tool"): conv.append({"role":r,"content":m.get("content") or ""})
        elif r=="assistant":
            tc=m.get("tool_calls") or []
            if not tc: continue
            conv.append({"role":"assistant","content":m.get("content") or "","tool_calls":tc})
    try:
        txt=TOK.apply_chat_template(conv, tools=ALFWORLD_TOOL_SCHEMAS, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        print("template fail",e, file=sys.stderr); break
    harvest(TOK.encode(txt, add_special_tokens=False), "TITOrun_traj")
    n+=1
    if n>=14: break
CTX={k:v for k,v in CTX.items() if len(v)>=8}
pickle.dump({f"{a}|{b}":v for (a,b),v in CTX.items()}, open("/workspace/tito-dressage/_probe_ctx.pkl","wb"))
print("CONTEXT SETS (fixed window = %d tokens, identical for every checkpoint):"%W)
for (a,b),v in sorted(CTX.items()): print(f"   {a:<24} {b:<14} n={len(v)}")
