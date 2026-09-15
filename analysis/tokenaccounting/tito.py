import json, glob, gzip, collections, re, os
D="/workspace/tito-dressage/examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories"
RE_REASON=re.compile(r"^Invalid tool call\..*?Reason: (.*?)(?:\n|$)", re.S)
def rcls(r):
    if r.startswith("missing <tool_call> block"): return "missing_block"
    if r.startswith("missing command argument"): return "missing_arg"
    if r.startswith("could not parse <tool_call> JSON"): return "bad_json"
    if r.startswith("expected env_step, got"): return "wrong_fname"
    if "is not in admissible commands" in r: return "not_admis"
    return "other"
files=sorted(glob.glob(D+"/*.jsonl.gz"))
print("TITO files:", [os.path.basename(f) for f in files])
bucket=collections.defaultdict(collections.Counter); rew=collections.defaultdict(list); gotv=collections.Counter()
for f in files:
    for line in gzip.open(f,'rt'):
        try: t=json.loads(line)
        except Exception: continue
        rid=t.get("rollout_id")
        if rid is None: continue
        it=int(rid)//128
        b=bucket[it]; rew[it].append(t.get("reward") or 0.0)
        for m in (t.get("messages") or []):
            if m.get("role")=="assistant": b["turns"]+=1
            elif m.get("role")=="tool":
                mm=RE_REASON.match(m.get("content") or "")
                if mm:
                    c=rcls(mm.group(1).strip()); b[c]+=1
                    if c=="wrong_fname": gotv[mm.group(1).split("got",1)[1].strip().strip("'\"")]+=1
K=["missing_block","wrong_fname","missing_arg","bad_json","not_admis"]
print(f"\n{'it':>4} {'trajs':>6} {'turns':>7} {'reward':>7} | " + " ".join(f"{k:>13}" for k in K))
for it in sorted(bucket):
    b=bucket[it]; n=b["turns"] or 1
    if len(rew[it])<20: continue
    print(f"{it:>4} {len(rew[it]):>6} {b['turns']:>7} {sum(rew[it])/len(rew[it]):>7.3f} | " + " ".join(f"{100*b[k]/n:>12.2f}%" for k in K))
print("\nTITO wrong-function-name values seen:", gotv.most_common(12) if gotv else "NONE AT ALL")
tot=collections.Counter()
for it in bucket:
    for k in K+["turns"]: tot[k]+=bucket[it][k]
print("\nTITO TOTALS: turns=%d  " % tot["turns"] + "  ".join(f"{k}={tot[k]} ({100*tot[k]/max(tot['turns'],1):.2f}%)" for k in K))
