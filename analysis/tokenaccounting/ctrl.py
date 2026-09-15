import json, glob, gzip, os, re, collections, sys
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
RE_CANON=re.compile(r"</think>\s*<tool_call>\s*<function=env_step>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>\s*<\|im_end\|>\s*$", re.S)
RE_REASON=re.compile(r"^Invalid tool call\..*?Reason: (.*?)(?:\n|$)", re.S)
def rcls(r):
    if r.startswith("missing <tool_call> block"): return "missing_block"
    if r.startswith("missing command argument"): return "missing_arg"
    if r.startswith("could not parse <tool_call> JSON"): return "bad_json"
    if r.startswith("expected env_step, got"): return "wrong_fname"
    if "is not in admissible commands" in r: return "not_admis"
    return "other"

def run(name, EXP, raw=True):
    idx={}
    if raw:
        for p in glob.glob(EXP+"/mimo_diff/*.json"):
            b=os.path.basename(p)[:-5]
            if "__" in b: idx[b.split("__",1)[1]]=p
    files=sorted(glob.glob(EXP+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0]))
    print("\n########", name, " rollout files:", len(files))
    print(f"{'it':>4} {'trajs':>6} {'turns':>6} {'reward':>7} {'CANON':>7} {'<<func':>7} {'badFn':>7} {'badPn':>7} | " +
          " ".join(f"{k:>10}" for k in ["missing_block","wrong_fname","missing_arg","not_admis"]))
    for f in files:
        it=int(f.split("_")[-1].split(".")[0])
        A=collections.Counter(); R=collections.Counter(); rew=[]; nt=0; ntr=0
        for line in gzip.open(f,'rt'):
            t=json.loads(line); ntr+=1; rew.append(t.get("reward") or 0.0)
            prev=None
            for m in (t.get("messages") or []):
                if m.get("role")=="assistant": A["mturns"]+=1
                elif m.get("role")=="tool":
                    mm=RE_REASON.match(m.get("content") or "")
                    if mm: R[rcls(mm.group(1).strip())]+=1
            if not raw: continue
            p=idx.get(t["session_id"])
            if not p: continue
            try: d=json.load(open(p))
            except Exception: continue
            tt=d.get("tito_tokens") or []; o=0; sl=[]
            for pt in (d.get("per_turn_diff") or []):
                la=pt.get("tito_len",0); sl.append(tt[o:o+la]); o+=la
            if not sl: continue
            for x in TOK.batch_decode(sl, skip_special_tokens=False):
                nt+=1
                if RE_CANON.search(x): A["canon"]+=1
                if "<<function=" in x: A["ddf"]+=1
                fns=[z for z in re.findall(r"<function=([^>\n]{0,60})>",x) if z!="env_step"]
                pns=[z for z in re.findall(r"<parameter=([^>\n]{0,60})>",x) if z!="command"]
                A["bf"]+=bool(fns); A["bp"]+=bool(pns)
        mt=A["mturns"] or 1; n=nt or 1
        pct=lambda v,d: f"{100*v/d:>6.1f}%"
        print(f"{it:>4} {ntr:>6} {nt if raw else A['mturns']:>6} {sum(rew)/max(len(rew),1):>7.3f} " +
              (f"{pct(A['canon'],n)} {pct(A['ddf'],n)} {pct(A['bf'],n)} {pct(A['bp'],n)} | " if raw else f"{'   n/a':>7} {'   n/a':>7} {'   n/a':>7} {'   n/a':>7} | ") +
              " ".join(f"{100*R[k]/mt:>9.2f}%" for k in ["missing_block","wrong_fname","missing_arg","not_admis"]))

run("CANONICAL MIMO #54a (mimo54_faithful)", "/workspace/tito-dressage/experiments/mimo54_faithful")
run("MIMO-100 (mimo_run_100)", "/workspace/tito-dressage/experiments/mimo_run_100")
