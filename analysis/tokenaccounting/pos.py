"""WHERE does the stray '<' land, and does landing position decide fatality?"""
import json, glob, gzip, os, re, collections
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
RE_REASON=re.compile(r"^Invalid tool call\.", re.S)
# position classes for an inserted '<'
P_SAFE=re.compile(r"<<+(?=(?:function=|parameter=|tool_call>))")   # extra '<' before a real tag word: TOLERATED
P_EQ  =re.compile(r"=<")                                          # '<' right after '=' : corrupts the NAME slot
P_SL  =re.compile(r"</<")                                         # '<' right after '/' : junk swallowed into COMMAND
P_ORPH=re.compile(r"<<+(?!(?:function=|parameter=|tool_call>|/))") # bracket run NOT completing a real tag
byk=collections.Counter(); byk_bad=collections.Counter()
per=collections.defaultdict(collections.Counter)
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    for line in gzip.open(f,'rt'):
        t=json.loads(line); p=idx.get(t["session_id"])
        if not p: continue
        try: d=json.load(open(p))
        except Exception: continue
        tt=d.get("tito_tokens") or []; o=0; sl=[]
        for pt in (d.get("per_turn_diff") or []):
            la=pt.get("tito_len",0); sl.append(tt[o:o+la]); o+=la
        if not sl: continue
        raws=TOK.batch_decode(sl, skip_special_tokens=False)
        rej=[]
        for m in (t.get("messages") or []):
            if m.get("role")=="assistant": rej.append(None)
            elif m.get("role")=="tool" and rej:
                rej[-1]=bool(RE_REASON.match(m.get("content") or ""))
        for i,x in enumerate(raws):
            if i>=len(rej) or rej[i] is None: continue
            emit=x.split("</think>")[-1] if "</think>" in x else x
            stray=sum(len(r)-1 for r in re.findall(r"<+", emit))
            k=min(stray,10)
            byk[k]+=1; byk_bad[k]+= 1 if rej[i] else 0
            c=per[it]; c["turns"]+=1
            c["safe"]+=len(P_SAFE.findall(emit)); c["eq"]+=len(P_EQ.findall(emit))
            c["sl"]+=len(P_SL.findall(emit));   c["orph"]+=len(P_ORPH.findall(emit))
            c["stray"]+=stray
print("A) REJECTION RATE vs NUMBER OF STRAY '<' IN THE TURN  (all 53 rounds pooled)")
print(f"{'stray <':>8} {'turns':>8} {'rejected':>9} {'reject rate':>12}")
for k in sorted(byk):
    lbl=f"{k}" if k<10 else "10+"
    print(f"{lbl:>8} {byk[k]:>8} {byk_bad[k]:>9} {100*byk_bad[k]/byk[k]:>11.1f}%")
print("\nB) WHERE THE STRAY '<' LANDS, per 100 turns")
print(f"{'it':>3} {'stray/turn':>11} | {'SAFE before tag':>16} {'FATAL after =':>14} {'FATAL after /':>14} {'FATAL orphan':>13} | {'fatal share':>12}")
for it in sorted(per):
    c=per[it]; n=c["turns"] or 1
    fat=c["eq"]+c["sl"]+c["orph"]; tot=fat+c["safe"]
    if it%4 and it<36: continue
    print(f"{it:>3} {c['stray']/n:>11.2f} | {100*c['safe']/n:>15.1f} {100*c['eq']/n:>14.1f} {100*c['sl']/n:>14.1f} {100*c['orph']/n:>13.1f} | {100*fat/max(tot,1):>11.1f}%")
