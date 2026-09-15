import json, glob, gzip, os, sys, collections, random
from transformers import AutoTokenizer
E="/workspace/tito-dressage/experiments/faithtext_tito"
OUT="/workspace/tito-dressage/_destab_dump"
os.makedirs(OUT, exist_ok=True)
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)

idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p

def trim_env(c, keep=6):
    if not c: return ""
    out=[]; incmd=False; n=0
    for ln in c.split("\n"):
        if ln.startswith("### Admissible Commands"): incmd=True; out.append(ln); continue
        if incmd and ln.startswith("- "):
            n+=1
            if n<=keep: out.append(ln)
            continue
        if incmd and not ln.startswith("- ") and ln.strip():
            if n>keep: out.append("- ...(%d more admissible commands)"%(n-keep))
            incmd=False; n=0
        out.append(ln)
    if incmd and n>keep: out.append("- ...(%d more admissible commands)"%(n-keep))
    return "\n".join(out)

def clip(s, lim=2600):
    if len(s)<=lim: return s
    return s[:lim//2] + "\n\n...[CLIPPED %d chars]...\n\n"%(len(s)-lim) + s[-lim//2:]

ITERS=[int(x) for x in sys.argv[1].split(",")]
PER=int(sys.argv[2])
written=0
for it in ITERS:
    f=E+"/trajectories/rollout_%d.jsonl.gz"%it
    if not os.path.exists(f): continue
    trajs=[json.loads(l) for l in gzip.open(f,'rt')]
    # stratify: distinct instance_id, prefer variety of task_type
    random.seed(1234+it)
    byinst=collections.defaultdict(list)
    for t in trajs: byinst[t.get("instance_id")].append(t)
    insts=sorted(byinst); random.shuffle(insts)
    picked=[]
    for i in insts:
        picked.append(random.choice(byinst[i]))
        if len(picked)>=PER: break
    for t in picked:
        sid=t["session_id"]; p=idx.get(sid)
        raws=[]
        if p:
            d=json.load(open(p)); tt=d.get("tito_tokens") or []; o=0
            for pt in (d.get("per_turn_diff") or []):
                la=pt.get("tito_len",0); raws.append(TOK.decode(tt[o:o+la], skip_special_tokens=False)); o+=la
        msgs=t.get("messages") or []
        name="it%02d__%s__%s.txt"%(it, t.get("task_type","?"), sid[4:12])
        L=[]
        L.append("="*100)
        L.append("ITERATION %d   session=%s"%(it,sid))
        L.append("instance_id=%s  task_type=%s"%(t.get("instance_id"), t.get("task_type")))
        L.append("reward=%s  task_success=%s  num_steps=%s  invalid_action_count=%s  truncated=%s  gen_finish_reason=%s"%(
            t.get("reward"),t.get("task_success"),t.get("num_steps"),t.get("invalid_action_count"),t.get("truncated"),t.get("gen_finish_reason")))
        L.append("response_length=%s origin_tokens_len=%s token_cap=%s compact_filtered=%s"%(
            t.get("response_length"),t.get("origin_tokens_len"),t.get("token_cap"),t.get("compact_filtered")))
        L.append("n_raw_turns=%d  n_messages=%d"%(len(raws),len(msgs)))
        L.append("="*100)
        # task
        for m in msgs[:2]:
            if m.get("role")=="user":
                L.append("\n### TASK PROMPT ###\n"+trim_env(m.get("content") or "", 4))
        ai=0
        for m in msgs:
            r=m.get("role")
            if r=="assistant":
                L.append("\n"+"-"*90)
                L.append(">>> ASSISTANT TURN %d  (RAW SAMPLED TEXT, verbatim) <<<"%ai)
                L.append("-"*90)
                L.append(clip(raws[ai]) if ai<len(raws) else "[NO RAW TEXT AVAILABLE]")
                tc=m.get("tool_calls")
                L.append("\n   [PARSED-> %s]"%(json.dumps(tc)[:300] if tc else "NO TOOL CALL PARSED"))
                ai+=1
            elif r=="tool":
                L.append("\n   --- ENV RESPONSE ---")
                L.append("   "+trim_env(m.get("content") or "",6).replace("\n","\n   "))
        open(os.path.join(OUT,name),"w").write("\n".join(L))
        written+=1
print("wrote",written,"files to",OUT)
