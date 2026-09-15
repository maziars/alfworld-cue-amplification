"""Exhaustive cause split for not_admissible (7134) and wrong_fname (7075) across all 53 iters."""
import json, glob, gzip, re, collections
E="/workspace/tito-dressage/experiments/faithtext_tito"
RE_NA=re.compile(r"Reason: command (.*) is not in admissible commands", re.S)
RE_WF=re.compile(r"Reason: expected env_step, got (.*?)(?:\n|$)", re.S)
TEMPLATE={"call","tool_name","tool_call","tool_calls","arguments","command","tool","tool_code",
          "tool_response","call_tool","example_function_name","tool_call_name","func_name",
          "function_name","tools","description","type","name","args","content","invoke","response"}
na=collections.defaultdict(collections.Counter); wf=collections.defaultdict(collections.Counter)
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    for line in gzip.open(f,'rt'):
        t=json.loads(line)
        for m in (t.get("messages") or []):
            if m.get("role")!="tool": continue
            c=m.get("content") or ""
            mm=RE_NA.search(c)
            if mm:
                v=mm.group(1).strip()
                na[it]["syntactic" if ("<" in v or "\\n" in v or "\n" in v) else "semantic"]+=1
                continue
            mm=RE_WF.search(c)
            if mm:
                v=mm.group(1).strip().strip("'\"")
                if "<" in v or "=" in v: k="nested_tag"
                elif v in TEMPLATE: k="template_word"
                elif re.fullmatch(r"[A-Za-z_]*step", v, re.I) or ("step" in v.lower()): k="env_step_typo"
                else: k="other_word"
                wf[it][k]+=1
print("not_admissible: SYNTACTIC (junk in command value) vs SEMANTIC (clean but wrong choice)")
print(f"{'it':>3} {'syntactic':>10} {'semantic':>9} {'%syn':>7}")
TS=TSEM=0
for it in sorted(na):
    c=na[it]; s,m=c["syntactic"],c["semantic"]; TS+=s; TSEM+=m
    if it%5 and it<36: continue
    print(f"{it:>3} {s:>10} {m:>9} {100*s/max(s+m,1):>6.1f}%")
print(f"TOT {TS:>10} {TSEM:>9} {100*TS/max(TS+TSEM,1):>6.1f}%")
print("\nwrong_fname cause split")
K=["nested_tag","template_word","env_step_typo","other_word"]
print(f"{'it':>3} " + " ".join(f"{k:>14}" for k in K))
tot=collections.Counter()
for it in sorted(wf):
    c=wf[it]
    for k in K: tot[k]+=c[k]
    if sum(c.values())<5: continue
    print(f"{it:>3} " + " ".join(f"{c[k]:>14}" for k in K))
print(f"TOT " + " ".join(f"{tot[k]:>14}" for k in K) + f"   total={sum(tot.values())}")
