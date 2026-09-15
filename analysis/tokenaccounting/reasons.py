import json, glob, gzip, collections, re
E="/workspace/tito-dressage/experiments/faithtext_tito"
RE=re.compile(r"Reason: (.*?)(?:\n|$)")
per=collections.defaultdict(collections.Counter)   # iter -> reasonclass -> n
gotvals=collections.Counter(); gotfirst={}
turns=collections.Counter()
def cls(r):
    if r.startswith("missing <tool_call> block"): return "missing_tool_call_block"
    if r.startswith("could not parse <tool_call> JSON"): return "unparseable_json"
    if r.startswith("expected env_step, got"): return "wrong_function_name"
    if "is not in admissible commands" in r: return "not_admissible"
    return "OTHER::"+r[:60]
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    for line in gzip.open(f,'rt'):
        t=json.loads(line)
        for m in (t.get("messages") or []):
            if m.get("role")=="assistant": turns[it]+=1
            if m.get("role")!="tool": continue
            mm=RE.search(m.get("content") or "")
            if not mm: continue
            r=mm.group(1).strip(); c=cls(r); per[it][c]+=1
            if c=="wrong_function_name":
                g=r.split("got",1)[1].strip().strip("'\"")
                gotvals[g]+=1; gotfirst.setdefault(g,it)
K=["missing_tool_call_block","wrong_function_name","unparseable_json","not_admissible"]
others=set()
for it in per:
    for k in per[it]:
        if k not in K: others.add(k)
print("ENV INVALID-REASON RATE PER ASSISTANT TURN (%), by iteration")
print(f"{'it':>3} {'turns':>6} " + " ".join(f"{k[:14]:>15}" for k in K))
for it in sorted(per):
    n=turns[it] or 1
    print(f"{it:>3} {turns[it]:>6} " + " ".join(f"{100*per[it][k]/n:>14.2f}%" for k in K))
print("\nOTHER reason classes:", sorted(others)[:10])
print("\nWRONG FUNCTION NAME -- what the model actually emitted (top 25):")
for g,c in gotvals.most_common(25):
    print(f"   {c:>6}  first_iter={gotfirst[g]:>3}  {g!r}")
print("   distinct wrong names:", len(gotvals))
