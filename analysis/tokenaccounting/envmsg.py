import json, glob, gzip, os, collections, re
E="/workspace/tito-dressage/experiments/faithtext_tito"
# Bottom-up census of ENV replies: cluster tool-message contents by their leading signature
sig_iter=collections.defaultdict(collections.Counter)   # sig -> iter -> count
examples={}
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    for line in gzip.open(f,'rt'):
        t=json.loads(line)
        for m in (t.get("messages") or []):
            if m.get("role")!="tool": continue
            c=(m.get("content") or "")
            head=c.split("\n",1)[0].strip()
            # normalize: strip quoted command content so clusters merge
            sig=re.sub(r"`[^`]*`","`X`",head)[:110]
            sig_iter[sig][it]+=1
            examples.setdefault(sig,c[:400])
print("DISTINCT ENV REPLY SIGNATURES:", len(sig_iter))
tot={s:sum(c.values()) for s,c in sig_iter.items()}
for s in sorted(tot, key=lambda x:-tot[x]):
    its=sig_iter[s]
    first=min(its); 
    print("\n### total=%d  first_iter=%d  |  %r"%(tot[s],first,s))
    band=lambda a,b: sum(v for k,v in its.items() if a<=k<=b)
    print("   by band: 1-10=%d 11-20=%d 21-29=%d 30-39=%d 40-41=%d 42-45=%d 46-53=%d"%(
        band(1,10),band(11,20),band(21,29),band(30,39),band(40,41),band(42,45),band(46,53)))
    if tot[s]<200000: print("   example:", repr(examples[s])[:300])
