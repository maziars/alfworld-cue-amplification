"""Same rollout, two tokenizations: TITO (exact sampled ids) vs faithful-text MIMO (re-encoded).
Text is IDENTICAL; only the chunking differs. Verify, then build paired probe contexts."""
import json, glob, os, torch, collections
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p

def runs_of_ones(mask):
    out=[];s=None
    for i,v in enumerate(mask):
        if v==1 and s is None: s=i
        elif v!=1 and s is not None: out.append((s,i)); s=None
    if s is not None: out.append((s,len(mask)))
    return out

def build_pair(sample):
    sid=sample["session_id"]; p=idx.get(sid)
    if not p: return None,"no mimo_diff"
    d=json.load(open(p))
    tito=d.get("tito_tokens") or []; per=d.get("per_turn_diff") or []
    toks=list(sample["tokens"]); lm=list(sample["loss_mask"])
    off=len(toks)-len(lm)
    spans=[(a+off,b+off) for a,b in runs_of_ones(lm)]
    if len(spans)!=len(per): return None,f"span/turn mismatch {len(spans)} vs {len(per)}"
    for (a,b),pt in zip(spans,per):
        if (b-a)!=pt.get("mimo_len"): return None,f"span len {b-a} != mimo_len {pt.get('mimo_len')}"
    # TITO version: replace each assistant span with its exact sampled ids
    out=[]; prev=0; o=0
    for (a,b),pt in zip(spans,per):
        la=pt.get("tito_len",0)
        out.extend(toks[prev:a]); out.extend(tito[o:o+la]); o+=la; prev=b
    out.extend(toks[prev:])
    if TOK.decode(out, skip_special_tokens=False)!=TOK.decode(toks, skip_special_tokens=False):
        return None,"TEXT MISMATCH"
    return (out, toks), "ok"

for it in (5,49):
    d=torch.load(f"{E}/debug/{it}.pt", map_location="cpu", weights_only=False)
    ok=bad=0; reasons=collections.Counter(); ex=None
    for s in d["samples"][:20]:
        r,msg=build_pair(s)
        if r: ok+=1; ex=ex or (len(r[0]),len(r[1]))
        else: bad+=1; reasons[msg]+=1
    print(f"round {it}: paired OK={ok} failed={bad} {dict(reasons)}  example lens tito={ex[0] if ex else '-'} mimo={ex[1] if ex else '-'}")
