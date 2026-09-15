"""Does the '<' accretion grow even in trajectories the reward RESTS on? If yes, reward exerts no corrective pressure."""
import json, glob, gzip, os, re, collections
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
print(f"{'it':>3} | {'REWARD=1: turns':>15} {'<<rate':>8} {'<perTurn':>9} | {'REWARD=0: turns':>15} {'<<rate':>8} {'<perTurn':>9}")
for it in [24,26,28,30,32,34,36,38,40,41,42]:
    f=E+"/trajectories/rollout_%d.jsonl.gz"%it
    S={1:collections.Counter(),0:collections.Counter()}
    for line in gzip.open(f,'rt'):
        t=json.loads(line); k=1 if (t.get("reward") or 0)>=0.5 else 0
        p=idx.get(t["session_id"])
        if not p: continue
        try: d=json.load(open(p))
        except Exception: continue
        tt=d.get("tito_tokens") or []; o=0; sl=[]
        for pt in (d.get("per_turn_diff") or []):
            la=pt.get("tito_len",0); sl.append(tt[o:o+la]); o+=la
        if not sl: continue
        for x in TOK.batch_decode(sl, skip_special_tokens=False):
            S[k]["n"]+=1
            if re.search(r"<<", x): S[k]["dd"]+=1
            S[k]["lt"]+=sum(len(m)-1 for m in re.findall(r"<+", x))
    out=[]
    for k in (1,0):
        c=S[k]; n=c["n"] or 1
        out.append(f"{c['n']:>15} {100*c['dd']/n:>7.1f}% {c['lt']/n:>9.2f}")
    print(f"{it:>3} | {out[0]} | {out[1]}")
