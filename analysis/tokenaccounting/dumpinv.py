"""Dump EVERY invalid env action's count, and a stratified sample WITH raw sampled text, per reason class."""
import json, glob, gzip, os, re, collections, random
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
OUT="/workspace/tito-dressage/_inv_dump"; os.makedirs(OUT, exist_ok=True)
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
RE_REASON=re.compile(r"^Invalid tool call\..*?Reason: (.*?)(?:\n|$)", re.S)
def rcls(r):
    if r.startswith("missing <tool_call> block"): return "missing_block"
    if r.startswith("missing command argument"): return "missing_arg"
    if r.startswith("could not parse <tool_call> JSON"): return "bad_json"
    if r.startswith("expected env_step, got"): return "wrong_fname"
    if "is not in admissible commands" in r: return "not_admissible"
    return "OTHER"
BANDS=[(1,10),(11,20),(21,27),(28,35),(36,39),(40,41),(42,43),(44,46),(47,50),(51,53)]
def band(it):
    for a,b in BANDS:
        if a<=it<=b: return f"{a}-{b}"
    return "?"
tot=collections.Counter(); per=collections.defaultdict(collections.Counter)
pool=collections.defaultdict(list)
random.seed(7)
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    for line in gzip.open(f,'rt'):
        t=json.loads(line); p=idx.get(t["session_id"])
        raws=None
        if p:
            try:
                d=json.load(open(p)); tt=d.get("tito_tokens") or []; o=0; sl=[]
                for pt in (d.get("per_turn_diff") or []):
                    la=pt.get("tito_len",0); sl.append(tt[o:o+la]); o+=la
                raws=TOK.batch_decode(sl, skip_special_tokens=False)
            except Exception: raws=None
        ai=-1; pend=None
        for m in (t.get("messages") or []):
            if m.get("role")=="assistant":
                ai+=1
                tc=m.get("tool_calls") or []
                pend=(ai, json.dumps(tc)[:400] if tc else None)
            elif m.get("role")=="tool" and pend is not None:
                mm=RE_REASON.match(m.get("content") or "")
                if mm:
                    reason=mm.group(1).strip(); c=rcls(reason)
                    tot[c]+=1; per[it][c]+=1
                    if raws and pend[0]<len(raws) and random.random()<0.02:
                        pool[c].append({"it":it,"band":band(it),"sid":t["session_id"][:12],"turn":pend[0],
                                        "reason":reason[:200],"parsed":pend[1],"raw":raws[pend[0]][:2000],
                                        "env_reply":(m.get("content") or "").split("### Admissible")[0][:400]})
                pend=None
print("TOTAL INVALID ENV ACTIONS BY REASON CLASS (all 53 iterations):")
g=sum(tot.values())
for c,n in tot.most_common(): print(f"   {n:>7} ({100*n/g:>5.2f}%)  {c}")
print(f"   {g:>7}  TOTAL")
# write per-class sample files, balanced across bands
for c,items in pool.items():
    byb=collections.defaultdict(list)
    for x in items: byb[x["band"]].append(x)
    sel=[]
    for b,_ in [(f"{a}-{bb}",0) for a,bb in BANDS]:
        random.shuffle(byb.get(b,[])); sel.extend(byb.get(b,[])[:7])
    L=[f"# INVALID-ACTION SAMPLE -- env reason class: {c}   (total in run: {tot[c]})",
       "# Each record: the RAW SAMPLED TEXT of the assistant turn the env rejected, what the parser extracted, and the env's reply.",""]
    for i,x in enumerate(sel):
        L.append("="*94)
        L.append(f"[{i}] iter={x['it']} band={x['band']} session={x['sid']} assistant_turn={x['turn']}")
        L.append(f"ENV REASON: {x['reason']}")
        L.append(f"PARSER EXTRACTED: {x['parsed']}")
        L.append("-"*40+" RAW SAMPLED TEXT (verbatim) "+"-"*40)
        L.append(x["raw"])
        L.append("-"*40+" ENV REPLY (head) "+"-"*40)
        L.append(x["env_reply"])
        L.append("")
    open(os.path.join(OUT,f"{c}.txt"),"w").write("\n".join(L))
    print(f"wrote {c}.txt with {len(sel)} records")
json.dump({"tot":dict(tot),"per":{str(k):dict(v) for k,v in per.items()}}, open("/workspace/tito-dressage/_inv_tot.json","w"))
