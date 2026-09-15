"""Probe contexts anchored to IDENTICAL TEXT on both ends.
For a probe at char p, pick c0 = a character boundary present in BOTH tokenizations,
>= TARGET_CHARS back. Context = the tokens spanning text[c0:p] in each tokenization.
Same characters, different chunking, different token counts."""
import json, glob, os, re, collections, pickle, torch
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
TARGET_CHARS=3000; PER=64; PER_TRAJ=2
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
    if TOK.decode(out,skip_special_tokens=False)!=TOK.decode(toks,skip_special_tokens=False): return None
    return out, toks, ts
CTX=collections.defaultdict(list); META=collections.defaultdict(list); DROP=collections.Counter()
def harvest(tito_ids, mimo_ids, tagT, tagM, traj, tspans):
    stT,txt=offs(tito_ids); stM,txt2=offs(mimo_ids)
    if txt!=txt2: DROP[("ALL","piecewise text differs")]+=1; return
    bT={p:k for k,p in enumerate(stT)}; bM={p:k for k,p in enumerate(stM)}
    shared=sorted(set(bT)&set(bM))                      # char offsets that are a boundary in BOTH
    import bisect
    cand=collections.defaultdict(list)
    for m in re.finditer(r"<(function|parameter)=", txt):
        p=m.start()
        for nm,o2 in (("TAG_START",0),("AFTER_ONE_LT",1),("AFTER_EQ",m.end()-p)): cand[nm].append(p+o2)
    for m in re.finditer(r"</(parameter|function|tool_call)>", txt):
        cand["AFTER_SLASH_"+("PARAM" if m.group(1)=="parameter" else "OTHER")].append(m.start()+2)
    for nm,ps in cand.items():
        usable=[]
        for p in sorted(set(ps)):
            if p not in bT or p not in bM: DROP[(nm,"end not a shared boundary")]+=1; continue
            j=bisect.bisect_right(shared, p-TARGET_CHARS)-1
            if j<0: DROP[(nm,"no room")]+=1; continue
            c0=shared[j]
            if p-c0 < TARGET_CHARS*0.6: DROP[(nm,"window too short")]+=1; continue
            usable.append((c0,p))
        if not usable: continue
        step=max(1,len(usable)//PER_TRAJ)
        for c0,p in usable[::step][:PER_TRAJ]:
            if len(CTX[(tagT,nm)])>=PER: break
            a,b=bT[c0],bT[p]; c,d2=bM[c0],bM[p]
            CTX[(tagT,nm)].append(tito_ids[a:b]); CTX[(tagM,nm)].append(mimo_ids[c:d2])
            turn=sum(1 for x,y in tspans if x<=b)
            META[(tagT,nm)].append({"traj":traj,"turn":turn,"chars":p-c0,"ntok_tito":b-a,"ntok_mimo":d2-c})
for it,tagT,tagM in ((5,"E1_r5_TITOtok","E2_r5_MIMOtok"),(49,"E3_r49_TITOtok","E4_r49_MIMOtok")):
    d=torch.load(f"{E}/debug/{it}.pt", map_location="cpu", weights_only=False)
    for si,s in enumerate(d["samples"][:70]):
        r=pair(s)
        if r: harvest(r[0], r[1], tagT, tagM, si, r[2])
CTX={k:v for k,v in CTX.items() if len(v)>=24}
pickle.dump({"ctx":{f"{a}|{b}":v for (a,b),v in CTX.items()},
             "meta":{f"{a}|{b}":v for (a,b),v in META.items()}},
            open("/workspace/tito-dressage/_probe_ctx.pkl","wb"))
print("TEXT-ALIGNED PAIRED CONTEXTS (identical characters on both ends; only chunking differs)")
for (a,b),v in sorted(CTX.items()):
    m=META.get((a,b)) or META.get((a.replace("E2","E1").replace("E4","E3"),b)) or []
    if m:
        tr=len({x['traj'] for x in m}); tu=sorted(x['turn'] for x in m)
        ch=sorted(x['chars'] for x in m); nt=sorted(x['ntok_tito'] for x in m); nm2=sorted(x['ntok_mimo'] for x in m)
        print(f"  {a:<18}{b:<20} n={len(v):<4} rollouts={tr:<3} turns {min(tu)}-{max(tu)} med {tu[len(tu)//2]} | chars med {ch[len(ch)//2]} | tokens tito med {nt[len(nt)//2]} mimo med {nm2[len(nm2)//2]}")
    else:
        print(f"  {a:<18}{b:<20} n={len(v):<4}")
print("\ndrops:")
for k,v in sorted(DROP.items()): print("   ",k,v)
