import json, glob, gzip, os, collections, statistics
E="/workspace/tito-dressage/experiments/faithtext_tito"
rows=[]
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    R=[];S=[];IA=[];TR=0;SU=0;FR=collections.Counter();n=0
    for line in gzip.open(f,'rt'):
        t=json.loads(line); n+=1
        R.append(t.get("reward") or 0.0); S.append(t.get("num_steps") or 0)
        IA.append(t.get("invalid_action_count") or 0); TR+=bool(t.get("truncated"))
        SU+=bool(t.get("task_success")); FR[t.get("gen_finish_reason")]+=1
    rows.append((it,n,statistics.mean(R),SU/n,statistics.mean(S),statistics.mean(IA),TR,dict(FR)))
print(f"{'it':>3} {'n':>4} {'reward':>7} {'succ':>6} {'steps':>6} {'inval':>7} {'trunc':>5}  finish_reasons")
for r in rows:
    print(f"{r[0]:>3} {r[1]:>4} {r[2]:>7.3f} {r[3]:>6.3f} {r[4]:>6.2f} {r[5]:>7.2f} {r[6]:>5}  {r[7]}")
