"""Probe contexts covering EXACTLY the same characters in both tokenizations.
Interior keeps each version's own chunking; the ragged ends are filled by re-encoding the
leftover text, so no probe point is ever dropped for lacking a shared token boundary."""
import json, glob, os, re, collections, pickle, bisect, torch
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
def slice_chars(ids, st, txt, c0, p):
    """Tokenization of exactly txt[c0:p]: natural tokens fully inside, ragged ends re-encoded."""
    n=len(ids)
    i=bisect.bisect_left(st, c0)                       # first token starting at/after c0
    j=i
    while j<n and st[j]+len(piece(ids[j]))<=p: j+=1    # last token ending at/before p
    if j<=i:                                           # no whole token fits
        return TOK.encode(txt[c0:p], add_special_tokens=False)
    head=TOK.encode(txt[c0:st[i]], add_special_tokens=False) if st[i]>c0 else []
    endc=st[j-1]+len(piece(ids[j-1]))
    tail=TOK.encode(txt[endc:p], add_special_tokens=False) if endc<p else []
    out=head+ids[i:j]+tail
    assert "".join(piece(t) for t in out)==txt[c0:p], "slice text mismatch"
    return out
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
CTX=collections.defaultdict(list); META=collections.defaultdict(list); DROP=collections.Counter(); SAME=collections.Counter()
def harvest(tito_ids, mimo_ids, tagT, tagM, traj, tspans):
    stT,txt=offs(tito_ids); stM,txt2=offs(mimo_ids)
    if txt!=txt2: DROP[("ALL","piecewise text differs")]+=1; return
    cand=collections.defaultdict(list)
    for m in re.finditer(r"<(function|parameter)=", txt):
        p=m.start()
        # require the '<' at p to be the FIRST of its run, so "after one '<'" means exactly one
        first_of_run = (p==0) or (txt[p-1]!="<")
        for nm,o2 in (("TAG_START",0),("AFTER_ONE_LT",1),("AFTER_EQ",m.end()-p)):
            if nm=="AFTER_ONE_LT" and not first_of_run:
                DROP[(nm,"not first bracket of its run")]+=1; continue
            if nm=="TAG_START" and not first_of_run:
                DROP[(nm,"not first bracket of its run")]+=1; continue
            cand[nm].append(p+o2)
    for m in re.finditer(r"</(parameter|function|tool_call)>", txt):
        cand["AFTER_SLASH_"+("PARAM" if m.group(1)=="parameter" else "OTHER")].append(m.start()+2)
    for nm,ps in cand.items():
        usable=[p for p in sorted(set(ps)) if p>=TARGET_CHARS]
        if not usable: DROP[(nm,"no room")]+=len(set(ps)); continue
        step=max(1,len(usable)//PER_TRAJ)
        for p in usable[::step][:PER_TRAJ]:
            if len(CTX[(tagT,nm)])>=PER: break
            c0=p-TARGET_CHARS
            try:
                a=slice_chars(tito_ids,stT,txt,c0,p); b=slice_chars(mimo_ids,stM,txt,c0,p)
            except AssertionError:
                DROP[(nm,"slice mismatch")]+=1; continue
            CTX[(tagT,nm)].append(a); CTX[(tagM,nm)].append(b)
            if a==b: SAME[nm]+=1
            turn=sum(1 for x,y in tspans if x<=bisect.bisect_left(stT,p))
            META[(tagT,nm)].append({"traj":traj,"turn":turn,"chars":p-c0,"ntok_tito":len(a),"ntok_mimo":len(b),
                                    "identical":int(a==b),"tail":txt[max(c0,p-24):p],"lt_run_before":len(txt[:p])-len(txt[:p].rstrip("<"))})
for it,tagT,tagM in ((5,"E1_r5_TITOtok","E2_r5_MIMOtok"),(49,"E3_r49_TITOtok","E4_r49_MIMOtok")):
    d=torch.load(f"{E}/debug/{it}.pt", map_location="cpu", weights_only=False)
    for si,s in enumerate(d["samples"][:70]):
        r=pair(s)
        if r: harvest(r[0], r[1], tagT, tagM, si, r[2])
CTX={k:v for k,v in CTX.items() if len(v)>=24}
pickle.dump({"ctx":{f"{a}|{b}":v for (a,b),v in CTX.items()},
             "meta":{f"{a}|{b}":v for (a,b),v in META.items()}},
            open("/workspace/tito-dressage/_probe_ctx.pkl","wb"))
print("EXACT-TEXT PAIRED CONTEXTS (same characters; ragged ends re-encoded; nothing dropped for boundaries)")
for (a,b),v in sorted(CTX.items()):
    m=META.get((a,b))
    if not m: print(f"  {a:<18}{b:<20} n={len(v)}"); continue
    tr=len({x['traj'] for x in m}); tu=sorted(x['turn'] for x in m)
    nt=sorted(x['ntok_tito'] for x in m); nm2=sorted(x['ntok_mimo'] for x in m); ident=sum(x['identical'] for x in m)
    print(f"  {a:<18}{b:<20} n={len(v):<4} rollouts={tr:<3} turns {min(tu)}-{max(tu)} med {tu[len(tu)//2]} | tok tito med {nt[len(nt)//2]} mimo med {nm2[len(nm2)//2]} | token-identical {ident}/{len(m)}")
print("\ndrops:", dict(DROP) or "none")
