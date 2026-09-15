"""How many DISTINCT rollouts actually contributed the 48 kept contexts?"""
import json, glob, os, re, collections, torch
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
W=1024; PER=48
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
_pc={}
def piece(i):
    s=_pc.get(i)
    if s is None: s=TOK.decode([i],skip_special_tokens=False); _pc[i]=s
    return s
def offs(ids):
    st=[];c=0
    for i in ids: st.append(c); c+=len(piece(i))
    return st,"".join(piece(i) for i in ids)
def runs1(m):
    o=[];s=None
    for i,v in enumerate(m):
        if v==1 and s is None: s=i
        elif v!=1 and s is not None: o.append((s,i)); s=None
    if s is not None: o.append((s,len(m)))
    return o
def pair(sample):
    p=idx.get(sample["session_id"])
    if not p: return None
    d=json.load(open(p)); tito=d.get("tito_tokens") or []; per=d.get("per_turn_diff") or []
    toks=list(sample["tokens"]); lm=list(sample["loss_mask"]); off=len(toks)-len(lm)
    spans=[(a+off,b+off) for a,b in runs1(lm)]
    if len(spans)!=len(per): return None
    for (a,b),pt in zip(spans,per):
        if (b-a)!=pt.get("mimo_len"): return None
    out=[];prev=0;o=0;ts=[]
    for (a,b),pt in zip(spans,per):
        la=pt.get("tito_len",0); out.extend(toks[prev:a]); s0=len(out); out.extend(tito[o:o+la]); ts.append((s0,len(out))); o+=la; prev=b
    out.extend(toks[prev:])
    return out, toks, ts
for it,tag in ((5,"healthy r5"),(49,"corrupt r49")):
    d=torch.load(f"{E}/debug/{it}.pt", map_location="cpu", weights_only=False)
    KEPT=collections.defaultdict(list)
    for si,s in enumerate(d["samples"][:20]):
        r=pair(s)
        if not r: continue
        tito_ids, mimo_ids, tspans = r
        stT,txt=offs(tito_ids); stM,_=offs(mimo_ids)
        bT={p:k for k,p in enumerate(stT)}; bM={p:k for k,p in enumerate(stM)}
        cand=collections.defaultdict(list)
        for m in re.finditer(r"<(function|parameter)=", txt):
            p=m.start()
            for nm,o2 in (("TAG_START",0),("AFTER_ONE_LT",1),("AFTER_EQ",m.end()-p)): cand[nm].append(p+o2)
        for m in re.finditer(r"</(parameter|function|tool_call)>", txt): cand["AFTER_SLASH"].append(m.start()+2)
        for nm,ps in cand.items():
            for p in sorted(set(ps)):
                kT=bT.get(p); kM=bM.get(p)
                if kT is None or kM is None or kT<W or kM<W: continue
                if len(KEPT[nm])>=PER: continue
                turn=sum(1 for a,b in tspans if a<=kT)
                KEPT[nm].append((si,turn))
    for nm,v in sorted(KEPT.items()):
        trajs=sorted({x[0] for x in v}); turns=sorted({x[1] for x in v})
        print(f"  {tag:<12} {nm:<14} kept={len(v):<3} from {len(trajs)} distinct rollout(s) {trajs[:6]}  turns {min(turns)}-{max(turns)}")
