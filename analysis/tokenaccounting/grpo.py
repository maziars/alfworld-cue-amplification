import json, glob, gzip, os, collections, statistics
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
print("GRPO GROUP STRUCTURE + TRAIN/SAMPLE LENGTH DIVERGENCE")
print(f"{'it':>3} {'grps':>5} {'allzero':>8} {'allone':>7} {'mixed':>7} {'zeroadv%':>9} | {'tito_len':>9} {'mimo_len':>9} {'shorter_by':>11} {'%':>7}")
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    g=collections.defaultdict(list); TL=0; ML=0
    for line in gzip.open(f,'rt'):
        t=json.loads(line)
        g[t.get("instance_id")].append(t.get("reward") or 0.0)
        p=idx.get(t["session_id"])
        if p:
            try:
                d=json.load(open(p)); td=d.get("trajectory_diff") or {}
                TL+=td.get("tito_len",0); ML+=td.get("mimo_len",0)
            except Exception: pass
    az=sum(1 for v in g.values() if max(v)==0); ao=sum(1 for v in g.values() if min(v)==1.0)
    mx=len(g)-az-ao
    print(f"{it:>3} {len(g):>5} {az:>8} {ao:>7} {mx:>7} {100*(az+ao)/max(len(g),1):>8.1f}% | {TL:>9} {ML:>9} {TL-ML:>11} {100*(TL-ML)/max(TL,1):>6.2f}%")
