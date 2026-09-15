"""For each FATAL construct: per-iteration count, first appearance, and whether the SAMPLED (tito)
and TRAINED (faithful-text re-encoded) token sequences differ AT THAT EXACT SPAN."""
import json, glob, gzip, os, re, collections, sys
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p

_pc={}
def piece(i):
    s=_pc.get(i)
    if s is None:
        s=TOK.decode([i], skip_special_tokens=False); _pc[i]=s
    return s
def offsets(ids):
    """char offset of each token; returns (list_of_starts, joined_text) or (None,None) if unreliable"""
    ps=[piece(i) for i in ids]; st=[]; c=0
    for s in ps: st.append(c); c+=len(s)
    return st, "".join(ps)
def toks_in_span(ids, st, txt, a, b):
    """token ids whose char range overlaps [a,b)"""
    out=[]
    for k,i in enumerate(ids):
        s=st[k]; e=s+len(piece(i))
        if e>a and s<b: out.append(i)
    return out

RE_REASON=re.compile(r"^Invalid tool call\..*?Reason: (.*?)(?:\n|$)", re.S)
def rcls(r):
    if r.startswith("missing <tool_call> block"): return "missing_block"
    if r.startswith("missing command argument"): return "missing_arg"
    if r.startswith("could not parse <tool_call> JSON"): return "bad_json"
    if r.startswith("expected env_step, got"): return "wrong_fname"
    if "is not in admissible commands" in r: return "not_admis"
    return "other"

PATS=[("F1_no_tool_call_opener", None),
      ("F2_tool_calls_plural", re.compile(r"<tool_calls>")),
      ("F3_dup_function_prefix", re.compile(r"<function=\s*<")),
      ("F4_broken_param_closer", re.compile(r"</\s*<")),
      ("F5_wrong_function_name", None)]

cnt=collections.defaultdict(collections.Counter)      # iter -> pattern -> n(turns, invalid only)
first={}                                              # pattern -> (iter, n)
tokdiff=collections.Counter(); toksame=collections.Counter()
tokdiff_it=collections.defaultdict(collections.Counter)
examples=collections.defaultdict(list)
badoff=0

its=sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(E+"/trajectories/rollout_*.jsonl.gz"))
for it in its:
    f=E+"/trajectories/rollout_%d.jsonl.gz"%it
    for line in gzip.open(f,'rt'):
        t=json.loads(line); p=idx.get(t["session_id"])
        if not p: continue
        try: d=json.load(open(p))
        except Exception: continue
        tt=d.get("tito_tokens") or []; mt=d.get("mimo_tokens") or []
        # per-turn reason + parsed command, walking messages
        reasons=[]; cmds=[]
        for m in (t.get("messages") or []):
            if m.get("role")=="assistant":
                reasons.append(None)
                tc=m.get("tool_calls") or []
                try: cmds.append(json.loads(tc[0]["function"]["arguments"]).get("command") if tc else None)
                except Exception: cmds.append(None)
            elif m.get("role")=="tool" and reasons:
                mm=RE_REASON.match(m.get("content") or "")
                reasons[-1]=rcls(mm.group(1).strip()) if mm else "ok"
        ot=om=0
        for ti,pt in enumerate(d.get("per_turn_diff") or []):
            la,lb=pt.get("tito_len",0),pt.get("mimo_len",0)
            A=tt[ot:ot+la]; B=mt[om:om+lb]; ot+=la; om+=lb
            if not A: continue
            r = reasons[ti] if ti < len(reasons) else None
            if r in (None,"ok"): continue          # only turns the env rejected
            sa,txta=offsets(A); sb,txtb=offsets(B)
            if txta!=txtb: badoff+=1; continue
            txt=txta
            emit = txt.split("</think>")[-1] if "</think>" in txt else txt
            base = len(txt)-len(emit)
            hits=[]
            if r=="missing_block" and "<tool_call>" not in txt:
                m2=re.search(r"<", emit)
                if m2: hits.append(("F1_no_tool_call_opener", base+m2.start(), base+m2.start()+1))
                else:  hits.append(("F1_no_tool_call_opener", base, base+1))
            for nm,rx in PATS:
                if rx is None: continue
                if nm=="F2_tool_calls_plural" and r!="missing_block": continue
                if nm=="F3_dup_function_prefix" and r!="wrong_fname": continue
                if nm=="F4_broken_param_closer":
                    c=cmds[ti] if ti<len(cmds) else None
                    if not (r=="not_admis" and c and ("<" in c or "\n" in c)): continue
                for mm2 in rx.finditer(emit):
                    hits.append((nm, base+mm2.start(), base+mm2.end())); break
            if r=="wrong_fname" and not any(h[0]=="F3_dup_function_prefix" for h in hits):
                mm2=re.search(r"<function=", emit)
                if mm2: hits.append(("F5_wrong_function_name", base+mm2.start(), base+mm2.end()+30))
            for nm,a,b in hits:
                cnt[it][nm]+=1
                if nm not in first: first[nm]=(it, txt[max(a-20,0):b+20])
                ta=toks_in_span(A,sa,txt,max(a-2,0),b+2); tb=toks_in_span(B,sb,txt,max(a-2,0),b+2)
                if ta!=tb:
                    tokdiff[nm]+=1; tokdiff_it[it][nm]+=1
                    if len(examples[nm])<3:
                        examples[nm].append((it, txt[max(a-12,0):b+12], ta[:10], tb[:10]))
                else: toksame[nm]+=1
NAMES=[n for n,_ in PATS]
print("skipped turns (offset check failed):", badoff)
print("\n=== A. FATAL CONSTRUCTS THAT PRODUCED AN INVALID ENV RESULT: turns per iteration ===")
print(f"{'it':>3} " + " ".join(f"{n.split('_',1)[0]:>6}" for n in NAMES) + f" {'total':>7}")
for it in its:
    c=cnt[it]
    if sum(c.values())==0 and it<20: continue
    print(f"{it:>3} " + " ".join(f"{c[n]:>6}" for n in NAMES) + f" {sum(c.values()):>7}")
print("\n=== B. FIRST APPEARANCE ===")
for n in NAMES:
    if n in first:
        it,ex=first[n]; print(f"  {n:<26} first invalid at it{it:<3}  {ex!r}")
    else: print(f"  {n:<26} never")
print("\n=== C. DOES THE CONSTRUCT TOKENIZE DIFFERENTLY? sampled(TITO) vs trained(faithful-text), same span ===")
print(f"{'construct':<26} {'differs':>9} {'same':>8} {'%differs':>9}")
for n in NAMES:
    dd,ss=tokdiff[n],toksame[n]; tot=dd+ss
    print(f"{n:<26} {dd:>9} {ss:>8} {100*dd/max(tot,1):>8.1f}%")
print("\n=== D. EXAMPLES (text | sampled ids | trained ids) ===")
for n in NAMES:
    for it,ex,ta,tb in examples[n][:2]:
        print(f"\n  {n} @it{it}\n    text   : {ex!r}\n    sampled: {ta}\n    trained: {tb}")
json.dump({"cnt":{str(k):dict(v) for k,v in cnt.items()},"tokdiff":dict(tokdiff),"toksame":dict(toksame),
           "tokdiff_it":{str(k):dict(v) for k,v in tokdiff_it.items()}}, open("/workspace/tito-dressage/_fatal_out.json","w"))
