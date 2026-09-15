import json, glob, gzip, os, re, collections, sys
from transformers import AutoTokenizer
E="/workspace/tito-dressage/experiments/faithtext_tito"
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p

RE_FUNC=re.compile(r"<function=([^>\n]{0,80})>")
RE_PARAM=re.compile(r"<parameter=([^>\n]{0,80})>")
RE_DUPCLOSE=re.compile(r"</parameter>\s*</parameter>")
RE_REASON=re.compile(r"^Invalid tool call\..*?Reason: (.*?)(?:\n|$)", re.S)
RE_LTRUN=re.compile(r"<+")
def rcls(r):
    if r.startswith("missing <tool_call> block"): return "missing_block"
    if r.startswith("missing command argument"): return "missing_arg"
    if r.startswith("could not parse <tool_call> JSON"): return "bad_json"
    if r.startswith("expected env_step, got"): return "wrong_fname"
    if "is not in admissible commands" in r: return "not_admissible"
    return "other"

# canonical healthy turn: prose </think> blank <tool_call> <function=env_step> <parameter=command> CMD </parameter> </function> </tool_call><|im_end|>
RE_CANON=re.compile(r"</think>\s*<tool_call>\s*<function=env_step>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>\s*<\|im_end\|>\s*$", re.S)

FEATS=["turns","canonical","has_tc_open","no_tc_open","dbl_lt_func","dbl_lt_param","dbl_close_slash",
       "dup_close_param","fname_bad","pname_bad","think_open","multi_think_close","tool_response",
       "no_im_end","ltrun_ge2","ltrun_ge3","nested_func","empty_after_think"]
agg=collections.defaultdict(collections.Counter)
reasons=collections.defaultdict(collections.Counter)
fname_ctr=collections.Counter(); pname_ctr=collections.Counter()
fname_first={}; pname_first={}
chars=collections.defaultdict(list)

its=sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(E+"/trajectories/rollout_*.jsonl.gz"))
for it in its:
    f=E+"/trajectories/rollout_%d.jsonl.gz"%it
    for line in gzip.open(f,'rt'):
        t=json.loads(line); p=idx.get(t["session_id"])
        if not p: continue
        try: d=json.load(open(p))
        except Exception: continue
        tt=d.get("tito_tokens") or []; o=0; slices=[]
        for pt in (d.get("per_turn_diff") or []):
            la=pt.get("tito_len",0); slices.append(tt[o:o+la]); o+=la
        if not slices: continue
        texts=TOK.batch_decode(slices, skip_special_tokens=False)
        # env reason per assistant turn (walk messages in order)
        msgs=t.get("messages") or []; per_turn_reason=[]; pend=None
        for m in msgs:
            if m.get("role")=="assistant": per_turn_reason.append(None)
            elif m.get("role")=="tool" and per_turn_reason:
                mm=RE_REASON.match(m.get("content") or "")
                per_turn_reason[-1]= rcls(mm.group(1).strip()) if mm else "ok"
        A=agg[it]
        for i,x in enumerate(texts):
            A["turns"]+=1; chars[it].append(len(x))
            body=x
            if RE_CANON.search(body): A["canonical"]+=1
            has_tc = "<tool_call>" in body
            A["has_tc_open"]+= has_tc; A["no_tc_open"]+= (not has_tc)
            A["dbl_lt_func"]+= body.count("<<function=")>0
            A["dbl_lt_param"]+= body.count("<<parameter=")>0
            A["dbl_close_slash"]+= body.count("</</")>0
            A["dup_close_param"]+= bool(RE_DUPCLOSE.search(body))
            fns=RE_FUNC.findall(body); pns=RE_PARAM.findall(body)
            bad_f=[z for z in fns if z!="env_step"]; bad_p=[z for z in pns if z!="command"]
            A["fname_bad"]+= bool(bad_f); A["pname_bad"]+= bool(bad_p)
            for z in bad_f:
                fname_ctr[z]+=1; fname_first.setdefault(z,it)
            for z in bad_p:
                pname_ctr[z]+=1; pname_first.setdefault(z,it)
            A["nested_func"]+= len(fns)>1
            A["think_open"]+= "<think>" in body
            A["multi_think_close"]+= body.count("</think>")>1
            A["tool_response"]+= "<tool_response>" in body
            A["no_im_end"]+= not body.rstrip().endswith("<|im_end|>")
            runs=[len(z) for z in RE_LTRUN.findall(body)] or [0]
            A["ltrun_ge2"]+= max(runs)>=2; A["ltrun_ge3"]+= max(runs)>=3
            tail=body.split("</think>")[-1] if "</think>" in body else body
            A["empty_after_think"]+= len(tail.strip().replace("<|im_end|>",""))==0
            r=per_turn_reason[i] if i<len(per_turn_reason) else None
            reasons[it][r or "none"]+=1
json.dump({"agg":{str(k):dict(v) for k,v in agg.items()},
           "reasons":{str(k):dict(v) for k,v in reasons.items()},
           "fname":dict(fname_ctr.most_common(300)),"fname_first":fname_first,
           "pname":dict(pname_ctr.most_common(300)),"pname_first":pname_first,
           "meanchars":{str(k):sum(v)/len(v) for k,v in chars.items()}},
          open("/workspace/tito-dressage/_scan_out.json","w"))
print("DONE. iters:",len(agg))
