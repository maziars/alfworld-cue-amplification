"""(a) drift rate in ACCEPTED vs REJECTED turns; (b) split F1 into prose-only vs structure-missing-opener."""
import json, glob, gzip, os, re, collections
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
RE_REASON=re.compile(r"^Invalid tool call\..*?Reason: (.*?)(?:\n|$)", re.S)
acc=collections.defaultdict(collections.Counter); f1=collections.defaultdict(collections.Counter)
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    for line in gzip.open(f,'rt'):
        t=json.loads(line); p=idx.get(t["session_id"])
        if not p: continue
        try: d=json.load(open(p))
        except Exception: continue
        tt=d.get("tito_tokens") or []
        reasons=[]
        for m in (t.get("messages") or []):
            if m.get("role")=="assistant": reasons.append(None)
            elif m.get("role")=="tool" and reasons:
                mm=RE_REASON.match(m.get("content") or "")
                reasons[-1]="bad" if mm else "ok"
        o=0; sl=[]
        for pt in (d.get("per_turn_diff") or []):
            la=pt.get("tito_len",0); sl.append((tt[o:o+la],pt)); o+=la
        txts=TOK.batch_decode([s for s,_ in sl], skip_special_tokens=False)
        for i,(x,(_,pt)) in enumerate(zip(txts,sl)):
            r=reasons[i] if i<len(reasons) else None
            if r is None: continue
            k="rej" if r=="bad" else "acc"
            acc[it][k]+=1
            if not pt.get("equal",True): acc[it][k+"_drift"]+=1
            if r=="bad" and "<tool_call>" not in x:
                emit=x.split("</think>")[-1] if "</think>" in x else x
                f1[it]["prose_only" if "<" not in emit.replace("<|im_end|>","") else "structure_no_opener"]+=1
print("(a) TURNS WITH BPE DRIFT: accepted vs rejected by the env")
print(f"{'it':>3} {'acc':>6} {'drift%':>8} | {'rej':>6} {'drift%':>8}")
for it in sorted(acc):
    c=acc[it]
    if it%4 and it<38: continue
    print(f"{it:>3} {c['acc']:>6} {100*c['acc_drift']/max(c['acc'],1):>7.1f}% | {c['rej']:>6} {100*c['rej_drift']/max(c['rej'],1):>7.1f}%")
print("\n(b) F1 SPLIT: prose-only turn (old mode) vs structure emitted but no <tool_call> opener (new mode)")
print(f"{'it':>3} {'prose_only':>11} {'structure_no_opener':>21}")
for it in sorted(f1):
    c=f1[it]
    if sum(c.values())<5 and it<38: continue
    print(f"{it:>3} {c['prose_only']:>11} {c['structure_no_opener']:>21}")
