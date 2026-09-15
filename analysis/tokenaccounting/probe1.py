import json, glob, gzip, os
E="/workspace/tito-dressage/experiments/faithtext_tito"
print("== dirs ==")
for d in sorted(os.listdir(E)):
    p=os.path.join(E,d)
    if os.path.isdir(p): print(" ",d, len(os.listdir(p)))
print("\n== rollout files ==")
rs=sorted(glob.glob(E+"/trajectories/*.jsonl.gz"))
print(len(rs), [os.path.basename(x) for x in rs[:5]], "...", [os.path.basename(x) for x in rs[-5:]])
# schema of one line
f=rs[-1]
with gzip.open(f,'rt') as fh:
    line=fh.readline()
d=json.loads(line)
print("\n== traj keys ==", sorted(d.keys()))
for k in ("session_id","reward","task_success","num_steps","invalid_action_count","truncated","gen_finish_reason","rollout_id"):
    if k in d: print("  ",k,"=",repr(d[k])[:120])
m=d.get("messages") or []
print("\n== n messages ==", len(m), "roles:", [x.get("role") for x in m][:12])
for i,x in enumerate(m[:6]):
    print(f"\n-- msg{i} role={x.get('role')} keys={sorted(x.keys())}")
    c=x.get("content")
    print("   content:", repr(c)[:600] if c else None)
    if x.get("tool_calls"): print("   tool_calls:", repr(x['tool_calls'])[:400])
