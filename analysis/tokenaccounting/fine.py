"""Fine-grained cause for EVERY one of the 50,480 rejected turns."""
import json, glob, gzip, os, re, collections
from transformers import AutoTokenizer
TOK=AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
E="/workspace/tito-dressage/experiments/faithtext_tito"
idx={}
for p in glob.glob(E+"/mimo_diff/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: idx[b.split("__",1)[1]]=p
RE_REASON=re.compile(r"^Invalid tool call\..*?Reason: (.*?)(?:\n|$)", re.S)
TEMPLATE={"call","tool_name","tool_call","tool_calls","arguments","command","tool","tool_code",
 "tool_response","call_tool","example_function_name","tool_call_name","func_name","function_name",
 "tools","description","type","name","args","content","invoke","response","parameter","parameters",
 "example_parameter_1","value","arg","arg_key","arg_value","function"}
def admissible(txt):
    out=[]
    if "### Admissible Commands" not in txt: return out
    for ln in txt.split("### Admissible Commands",1)[1].split("\n"):
        if ln.startswith("- "): out.append(ln[2:].strip())
        elif ln.strip() and out: break
    return out
def classify(reason, emit, parsed_name, parsed_cmd, adm):
    # ---- no structure at all
    core=emit.replace("<|im_end|>","")
    if "<" not in core:
        return ("A1_prose_no_action" if re.search(r"task (is|has been|was)?\s*(complete|completed|done|finished)", core, re.I) else "A2_prose_other")
    # ---- degeneration
    if re.search(r"(?:<\s*){12,}", emit): return "G1_bracket_storm"
    if re.search(r"(<tool\b|<function\b|<parameter=command>|<tool_name\b|<arg_value>)(?:[^<]{0,3}\1){7,}", emit): return "G2_token_loop"
    if not emit.rstrip().endswith("<|im_end|>"): return "G3_truncated_midgen"
    # ---- name-slot faults
    if reason=="wrong_fname":
        n=parsed_name or ""
        if "<" in n or "=" in n: return "C1_name_slot_ate_a_tag"
        if n in TEMPLATE: return "C2_name_is_template_word"
        if "step" in n.lower(): return "C3_name_misspelled"
        return "C4_name_other_word"
    # ---- parameter-slot faults
    if reason=="missing_arg":
        if re.search(r"<+function=(command|arguments|tool_name)\b", emit): return "D1_param_keyword_swapped_to_function"
        if re.search(r"<+(command|arg|arg_key|arg_value|args)[>=]", emit): return "D2_param_tag_wrong_word"
        if re.search(r"<+parameter=(?!command>)[a-z_]+>", emit): return "D3_param_name_wrong"
        if re.search(r"<+parameter=command>\s*</", emit): return "D5_param_value_empty"
        return "D4_param_other"
    # ---- value contamination vs genuine bad choice
    if reason=="not_admissible":
        c=parsed_cmd or ""
        if "<" in c or "\n" in c: return "E1_closer_junk_in_command"
        if not adm: return "F5_semantic_other"
        toks=[w for w in re.findall(r"[a-z]+", c.lower()) if w not in ("go","to","take","from","put","in","on","open","close","examine","use","move","clean","heat","cool","with","look","inventory")]
        obj=toks[0] if toks else ""
        anywhere=any(obj and obj in a for a in adm)
        if c.startswith("go to") and anywhere: return "F4_goto_where_already_standing"
        if not anywhere: return "F2_object_not_in_this_room"
        return "F1_object_exists_wrong_precondition"
    # ---- block-level faults
    if reason=="bad_json": return "B4_tagpair_content_unparseable"
    if reason=="missing_block":
        if re.search(r"<+/?(tool_calls|tool_name|tool_reference|tool)\b[>=]", emit) and "<tool_call>" not in emit: return "B2_opener_word_corrupted"
        if re.search(r"<+/?(function|parameter)[>=]", emit) and "<tool_call>" not in emit: return "B1_opener_omitted_body_present"
        if "<tool_call>" in emit: return "B3_opener_present_body_unusable"
        return "B5_other_no_block"
    return "Z_unclassified"
cnt=collections.defaultdict(collections.Counter); ex=collections.defaultdict(list)
for f in sorted(glob.glob(E+"/trajectories/rollout_*.jsonl.gz"), key=lambda p:int(p.split("_")[-1].split(".")[0])):
    it=int(f.split("_")[-1].split(".")[0])
    for line in gzip.open(f,'rt'):
        t=json.loads(line); p=idx.get(t["session_id"])
        if not p: continue
        try: d=json.load(open(p))
        except Exception: continue
        tt=d.get("tito_tokens") or []; o=0; sl=[]
        for pt in (d.get("per_turn_diff") or []):
            la=pt.get("tito_len",0); sl.append(tt[o:o+la]); o+=la
        if not sl: continue
        raws=TOK.batch_decode(sl, skip_special_tokens=False)
        ai=-1; pend=None
        for m in (t.get("messages") or []):
            if m.get("role")=="assistant":
                ai+=1; tc=m.get("tool_calls") or []
                nm=cm=None
                if tc:
                    try:
                        nm=tc[0]["function"]["name"]; cm=json.loads(tc[0]["function"]["arguments"]).get("command")
                    except Exception: pass
                pend=(ai,nm,cm)
            elif m.get("role")=="tool" and pend is not None:
                c=m.get("content") or ""; mm=RE_REASON.match(c)
                if mm and pend[0]<len(raws):
                    r=mm.group(1).strip()
                    rc=("missing_block" if r.startswith("missing <tool_call>") else
                        "missing_arg" if r.startswith("missing command") else
                        "bad_json" if r.startswith("could not parse") else
                        "wrong_fname" if r.startswith("expected env_step") else
                        "not_admissible" if "not in admissible" in r else "other")
                    x=raws[pend[0]]; emit=x.split("</think>")[-1] if "</think>" in x else x
                    k=classify(rc, emit, pend[1], pend[2], admissible(c))
                    cnt[it][k]+=1
                    if len(ex[k])<4: ex[k].append((it, emit[:300]))
                pend=None
ALL=sorted({k for c in cnt.values() for k in c})
tot=collections.Counter()
for c in cnt.values():
    for k,v in c.items(): tot[k]+=v
print("FINE-GRAINED CAUSE OF EVERY REJECTED TURN  (total %d)"%sum(tot.values()))
for k,v in tot.most_common(): print(f"   {v:>7} ({100*v/sum(tot.values()):>5.2f}%)  {k}")
json.dump({"cnt":{str(k):dict(v) for k,v in cnt.items()},"tot":dict(tot),
           "ex":{k:v for k,v in ex.items()}}, open("/workspace/tito-dressage/_fine_out.json","w"))
print("\nPER-ITERATION (top causes)")
TOPK=[k for k,_ in tot.most_common(12)]
print(f"{'it':>3} " + " ".join(f"{k[:11]:>12}" for k in TOPK))
for it in sorted(cnt):
    c=cnt[it]
    print(f"{it:>3} " + " ".join(f"{c[k]:>12}" for k in TOPK))
