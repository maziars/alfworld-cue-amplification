"""Probe contexts from the SAME rollouts, tokenized two ways.
E1/E2 = round 5 (clean) as TITO ids vs faithful-text ids.  E3/E4 = round 49 (corrupted), same pairing.
Probe positions are the SAME CHARACTER OFFSET in both members of a pair."""
import json, glob, gzip, os, re, collections, pickle, sys, torch
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
W=1024; PER=64; PER_TRAJ=2
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
_pc={}
def piece(i):
    s=_pc.get(i)
    if s is None: s=TOK.decode([i], skip_special_tokens=False); _pc[i]=s
    return s
def offs(ids):
    st=[];c=0
    for i in ids: st.append(c); c+=len(piece(i))
    return st,"".join(piece(i) for i in ids)
def runs1(mask):
    out=[];s=None
    for i,v in enumerate(mask):
        if v==1 and s is None: s=i
        elif v!=1 and s is not None: out.append((s,i)); s=None
    if s is not None: out.append((s,len(mask)))
    return out
def pair(sample):
    p=idx.get(sample["session_id"])
    if not p: return None
    d=json.load(open(p)); tito=d.get("tito_tokens") or []; per=d.get("per_turn_diff") or []
    toks=list(sample["tokens"]); lm=list(sample["loss_mask"]); off=len(toks)-len(lm)
    spans=[(a+off,b+off) for a,b in runs1(lm)]
    if len(spans)!=len(per): return None
    for (a,b),pt in zip(spans,per):
        if (b-a)!=pt.get("mimo_len"): return None
    out=[]; prev=0; o=0; ts=[]
    for (a,b),pt in zip(spans,per):
        la=pt.get("tito_len",0); out.extend(toks[prev:a]); s0=len(out); out.extend(tito[o:o+la]); ts.append((s0,len(out))); o+=la; prev=b
    out.extend(toks[prev:])
    TSPANS[0]=ts
    if TOK.decode(out,skip_special_tokens=False)!=TOK.decode(toks,skip_special_tokens=False): return None
    return out, toks
CTX=collections.defaultdict(list); DROP=collections.Counter()
PROV=collections.defaultdict(list); TRAJ_ID=[0]; TSPANS=[[]]
def harvest_pair(tito_ids, mimo_ids, tagT, tagM):
    stT,txt=offs(tito_ids); stM,txt2=offs(mimo_ids)
    if txt!=txt2:
        DROP[("ALL","piecewise text differs (split multibyte char)")]+=1
        return
    bT={p:k for k,p in enumerate(stT)}; bM={p:k for k,p in enumerate(stM)}
    cand=collections.defaultdict(list)
    for m in re.finditer(r"<(function|parameter)=", txt):
        p=m.start()
        for nm,o2 in (("TAG_START",0),("AFTER_ONE_LT",1),("AFTER_EQ",m.end()-p)): cand[nm].append(p+o2)
    for m in re.finditer(r"</(parameter|function|tool_call)>", txt): cand["AFTER_SLASH"].append(m.start()+2)
    for nm,ps in cand.items():
        usable=[]
        for p in sorted(set(ps)):
            kT=bT.get(p); kM=bM.get(p)
            if kT is None or kM is None: DROP[(nm,"no shared boundary")]+=1; continue
            if kT<W or kM<W: DROP[(nm,"too early")]+=1; continue
            usable.append((kT,kM,p))
        if not usable: continue
        # spread evenly through THIS trajectory, at most PER_TRAJ from it
        step=max(1,len(usable)//PER_TRAJ)
        for kT,kM,p in usable[::step][:PER_TRAJ]:
            if len(CTX[(tagT,nm)])>=PER: break
            CTX[(tagT,nm)].append(tito_ids[kT-W:kT]); CTX[(tagM,nm)].append(mimo_ids[kM-W:kM])
            PROV[(tagT,nm)].append((TRAJ_ID[0], sum(1 for a,b in TSPANS[0] if a<=kT)))
for it,tagT,tagM in ((5,"E1_r5_TITOtok","E2_r5_MIMOtok"),(49,"E3_r49_TITOtok","E4_r49_MIMOtok")):
    d=torch.load(f"{E}/debug/{it}.pt", map_location="cpu", weights_only=False)
    for si,s in enumerate(d["samples"][:60]):
        r=pair(s)
        if r:
            TRAJ_ID[0]=si
            harvest_pair(r[0], r[1], tagT, tagM)
# external reference: the real TITO run's own trajectories
ref=TOK.decode(list(torch.load(f"{E}/debug/5.pt",map_location="cpu",weights_only=False)["samples"][0]["tokens"]),skip_special_tokens=False)
SYS=ref[:ref.index("<|im_end|>")+len("<|im_end|>")+1]
TJ="/workspace/tito-dressage/examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories/rollout_1.jsonl.gz"
n=0
for line in gzip.open(TJ,'rt'):
    t=json.loads(line); msgs=t.get("messages") or []
    if len(msgs)<12: continue
    s=SYS
    for mm in msgs:
        r=mm.get("role")
        if r=="system": continue
        if r in ("user","tool"): s+="<|im_start|>user\n"+(mm.get("content") or "")+"<|im_end|>\n"
        elif r=="assistant":
            tc=mm.get("tool_calls") or []
            if not tc: continue
            try: cmd=json.loads(tc[0]["function"]["arguments"]).get("command","")
            except Exception: continue
            body=(mm.get("reasoning_content") or "Choosing the next admissible command.").strip()
            s+=("<|im_start|>assistant\n<think>\n"+body+"\n</think>\n\n<tool_call>\n<function=env_step>\n"
                "<parameter=command>\n"+cmd+"\n</parameter>\n</function>\n</tool_call><|im_end|>\n")
    ids=TOK.encode(s, add_special_tokens=False)
    harvest_pair(ids, ids, "B_TITOrun_traj", "B_TITOrun_traj")
    n+=1
    if n>=16: break
CTX={k:v for k,v in CTX.items() if len(v)>=16}
pickle.dump({f"{a}|{b}":v for (a,b),v in CTX.items()}, open("/workspace/tito-dressage/_probe_ctx.pkl","wb"))
print("PAIRED CONTEXT SETS (identical text + identical character position; only the chunking differs):")
for (a,b),v in sorted(CTX.items()): print(f"   {a:<18} {b:<14} n={len(v)}")
print("\nPROVENANCE of kept contexts:")
for k in sorted(PROV):
    v=PROV[k]; tr=sorted({x[0] for x in v}); tu=sorted(x[1] for x in v)
    print(f"   {k[0]:<18}{k[1]:<14} n={len(v):<4} rollouts={len(tr):<3} turns {min(tu)}-{max(tu)} median {tu[len(tu)//2]}")
print("\ndropped probe points (no shared token boundary => not comparable):")
for k,v in sorted(DROP.items()): print("   ",k,v)
