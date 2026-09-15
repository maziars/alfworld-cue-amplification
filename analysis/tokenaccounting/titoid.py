import json, glob, gzip, collections, os
D="/workspace/tito-dressage/examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories"
for f in sorted(glob.glob(D+"/*.jsonl.gz")):
    n=0; rid=[]; rw=collections.Counter(); succ=0; keys=None
    for line in gzip.open(f,'rt'):
        t=json.loads(line); n+=1
        if keys is None: keys=sorted(t.keys())
        rid.append(t.get("rollout_id") or 0); rw[round(t.get("reward") or 0,2)]+=1; succ+=bool(t.get("task_success"))
    print(f"{os.path.basename(f):>22} n={n:>6} rollout_id={min(rid)}..{max(rid)} success={succ}/{n} ({100*succ/n:.1f}%) reward_hist={dict(rw.most_common(5))}")
print("\nkeys:", keys)
print("\nmtimes:")
for f in sorted(glob.glob(D+"/*.jsonl.gz")): print("  ",os.path.basename(f), os.path.getmtime(f))
