"""Fixed probe contexts, kept SEPARATE by trajectory source. Identical windows reused for every checkpoint."""
import json, glob, gzip, os, re, collections, pickle, sys
import torch
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
W=1024; PER=48
_pc={}
def piece(i):
    s=_pc.get(i)
    if s is None: s=TOK.decode([i], skip_special_tokens=False); _pc[i]=s
    return s
def offsets(ids):
    st=[];c=0
    for i in ids: st.append(c); c+=len(piece(i))
    return st,"".join(piece(i) for i in ids)
CTX=collections.defaultdict(list)
def harvest(ids, src):
    st,txt=offsets(ids); b2i={p:k for k,p in enumerate(st)}
    hits=collections.defaultdict(list)
    for m in re.finditer(r"<(function|parameter)=", txt):
        p=m.start()
        for nm,off in (("TAG_START",0),("AFTER_ONE_LT",1),("AFTER_EQ",m.end()-p)):
            k=b2i.get(p+off)
            if k and k>=W: hits[nm].append(k)
    for m in re.finditer(r"</(parameter|function|tool_call)>", txt):
        k=b2i.get(m.start()+2)
        if k and k>=W: hits["AFTER_SLASH"].append(k)
    for nm,ks in hits.items():
        for k in sorted(set(ks)):
            if len(CTX[(src,nm)])<PER: CTX[(src,nm)].append(ids[k-W:k])
def from_debug(pt, src, ntraj=14):
    try: d=torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as e: print("skip",pt,e,file=sys.stderr); return
    for s in d["samples"][:ntraj]: harvest(list(s["tokens"]), src)
FT="/workspace/tito-dressage/experiments/faithtext_tito/debug"
from_debug(f"{FT}/5.pt","A_faithtext_r5_clean")
from_debug(f"{FT}/49.pt","D_faithtext_r49_corrupt")
f=sorted(glob.glob("/workspace/tito-dressage/experiments/mimo54_faithful/debug/*.pt"))
if f: from_debug(f[0],"C_canonicalMIMO_traj")
f=sorted(glob.glob("/workspace/tito-dressage/experiments/mimo_run_100/debug/*.pt"))
if f: from_debug(f[0],"C2_mimo100_traj")

# ---- REAL TITO-run trajectories, rendered into the run's own chat scaffold
d=torch.load(f"{FT}/5.pt", map_location="cpu", weights_only=False)
ref=TOK.decode(list(d["samples"][0]["tokens"]), skip_special_tokens=False)
SYS=ref[:ref.index("<|im_end|>")+len("<|im_end|>")+1]          # system block incl. trailing newline
m=re.search(r"<\|im_start\|>assistant\n(.{0,40})", ref, re.S)
APFX=m.group(1)[:m.group(1).find("\n")+1] if m else "<think>\n"
print("assistant prefix after role marker:", repr(APFX), file=sys.stderr)
TJ="/workspace/tito-dressage/examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories/rollout_1.jsonl.gz"
n=0
for line in gzip.open(TJ,'rt'):
    t=json.loads(line); msgs=t.get("messages") or []
    if len(msgs)<12: continue
    s=SYS
    for mm in msgs:
        r=mm.get("role")
        if r=="system": continue
        if r in ("user","tool"):
            s+="<|im_start|>user\n"+(mm.get("content") or "")+"<|im_end|>\n"
        elif r=="assistant":
            tc=mm.get("tool_calls") or []
            if not tc: continue
            try: cmd=json.loads(tc[0]["function"]["arguments"]).get("command","")
            except Exception: continue
            body=(mm.get("reasoning_content") or "Choosing the next admissible command.").strip()
            s+=("<|im_start|>assistant\n"+APFX+body+"\n</think>\n\n<tool_call>\n<function=env_step>\n"
                "<parameter=command>\n"+cmd+"\n</parameter>\n</function>\n</tool_call><|im_end|>\n")
    harvest(TOK.encode(s, add_special_tokens=False), "B_TITOrun_traj")
    n+=1
    if n>=16: break
CTX={k:v for k,v in CTX.items() if len(v)>=16}
pickle.dump({f"{a}|{b}":v for (a,b),v in CTX.items()}, open("/workspace/tito-dressage/_probe_ctx.pkl","wb"))
print("FIXED CONTEXT SETS (window=%d tokens, identical across all checkpoints):"%W)
for (a,b),v in sorted(CTX.items()): print(f"   {a:<26} {b:<14} n={len(v)}")
