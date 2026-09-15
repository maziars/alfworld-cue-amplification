"""Where does the faithful-text BPE drift actually land? tito_tokens = SAMPLED ids, mimo_tokens = TRAINED ids (same text)."""
import json, glob, os, collections
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
import gzip
sampled_only=collections.Counter(); trained_only=collections.Counter()
per_it=collections.defaultdict(lambda: collections.Counter())
for it in [5,20,28,30,34,38,40,42,46,50,53]:
    f=E+"/trajectories/rollout_%d.jsonl.gz"%it
    if not os.path.exists(f): continue
    for line in gzip.open(f,'rt'):
        t=json.loads(line); p=idx.get(t["session_id"])
        if not p: continue
        try: d=json.load(open(p))
        except Exception: continue
        tt=d.get("tito_tokens") or []; mt=d.get("mimo_tokens") or []
        ot=om=0
        for pt in (d.get("per_turn_diff") or []):
            la,lb=pt.get("tito_len",0),pt.get("mimo_len",0)
            a,b=tt[ot:ot+la],mt[om:om+lb]; ot+=la; om+=lb
            per_it[it]["turns"]+=1
            if pt.get("equal",True): continue
            per_it[it]["drift_turns"]+=1
            ca,cb=collections.Counter(a),collections.Counter(b)
            do=ca-cb; dn=cb-ca
            for tokid,c in do.items():
                sampled_only[tokid]+=c; per_it[it]["ntok"]+=c
            for tokid,c in dn.items(): trained_only[tokid]+=c
def show(name,ctr):
    tot=sum(ctr.values())
    print(f"\n{name}  (total {tot} token-instances, {len(ctr)} distinct ids)")
    lt=0
    for tid,c in ctr.most_common(14):
        s=TOK.decode([tid]); 
        if set(s)<= set("<"): lt+=c
        print(f"   {c:>7} ({100*c/max(tot,1):>5.2f}%)  id={tid:<7} {s!r}")
    allang=sum(c for tid,c in ctr.items() if set(TOK.decode([tid]))<=set("<") and TOK.decode([tid]))
    print(f"   --> tokens made ONLY of '<' characters: {allang} = {100*allang/max(tot,1):.2f}% of all drift")
show("SAMPLED-side tokens that the TRAINED sequence did NOT contain", sampled_only)
show("TRAINED-side tokens that the SAMPLED sequence did NOT contain", trained_only)
print("\nDRIFT RATE BY ITERATION")
print(f"{'it':>4} {'turns':>7} {'drift_turns':>12} {'%drift':>8} {'drift_tok/turn':>15}")
for it in sorted(per_it):
    c=per_it[it]; n=c["turns"] or 1
    print(f"{it:>4} {c['turns']:>7} {c['drift_turns']:>12} {100*c['drift_turns']/n:>7.1f}% {c['ntok']/n:>15.3f}")
