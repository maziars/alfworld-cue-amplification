"""Per-context probe. Saves EVERY individual probability so results can be re-analysed later.
Contexts are variable length (text-aligned), so batch=1 with logits_to_keep=1."""
import json, sys, collections, pickle, statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL=sys.argv[1]; OUT=sys.argv[2]; TAG=sys.argv[3]
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
BR=collections.defaultdict(list)
for i in range(len(TOK)):
    s=TOK.convert_ids_to_tokens(i)
    if not s: continue
    d=TOK.convert_tokens_to_string([s])
    if d and set(d)=={"<"}: BR[len(d)].append(i)
def tid(s):
    e=TOK.encode(s, add_special_tokens=False); return e[0] if len(e)==1 else None
WORD={w:tid(w) for w in ["env_step","function","parameter","command"]}
P=pickle.load(open("/workspace/tito-dressage/_probe_ctx.pkl","rb"))
CTX=P["ctx"]; META=P["meta"]
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
REC=[]
with torch.no_grad():
    for key,ctxs in sorted(CTX.items()):
        mk=key if key in META else key.replace("E2_r5_MIMOtok","E1_r5_TITOtok").replace("E4_r49_MIMOtok","E3_r49_TITOtok")
        meta=META.get(mk) or [{}]*len(ctxs)
        for i,c in enumerate(ctxs):
            if len(c)<16: continue
            ii=torch.tensor([c], device="cuda")
            try: lg=model(ii, logits_to_keep=1).logits
            except TypeError: lg=model(ii).logits
            p=torch.softmax(lg[0,-1,:].float(), -1)
            r={"set":key.split("|")[0],"pos":key.split("|")[1],"i":i,"ntok":len(c)}
            r.update({k:v for k,v in (meta[i] if i<len(meta) else {}).items()})
            tot=0.0
            for n,ids in sorted(BR.items()):
                v=float(p[ids].sum()); r[f"p_lt{n}"]=v
                if n>=2: tot+=v
            r["p_multi"]=tot
            for w,t in WORD.items():
                if t is not None: r[f"p_{w}"]=float(p[t])
            tp=torch.topk(p,5)
            r["top5"]=[[int(a), TOK.decode([int(a)], skip_special_tokens=False), float(b)] for a,b in zip(tp.indices,tp.values)]
            REC.append(r)
json.dump({"tag":TAG,"model":MODEL,"records":REC}, open(OUT,"w"))
# summary: mean AND median AND quartiles, so no single statistic is load-bearing
G=collections.defaultdict(list)
for r in REC: G[(r["set"],r["pos"])].append(r)
def q(v,f):
    v=sorted(v); return v[min(len(v)-1,int(f*len(v)))]
for k,v in sorted(G.items()):
    for fld in ("p_lt1","p_multi"):
        x=[r[fld] for r in v]
        print(f"{TAG} {k[0]:<16} {k[1]:<18} {fld:<8} n={len(x):<3} mean={statistics.mean(x):.5f} median={statistics.median(x):.5f} p25={q(x,.25):.5f} p75={q(x,.75):.5f} max={max(x):.5f}")
