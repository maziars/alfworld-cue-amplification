#!/usr/bin/env python3
"""Decode the RARE content substitutions (turns with n_sub>0) — the 'token id
changes' on top of the universal boundary shift.
Usage: decode_subs.py <mimo_diff_dir> <tokenizer_path>"""
import json, sys, glob, os, difflib
diffdir, tokpath = sys.argv[1], sys.argv[2]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tokpath, trust_remote_code=True)

def spans(flat, lens):
    o, out = 0, []
    for L in lens:
        out.append(flat[o:o + L]); o += L
    return out

shown = 0
nfiles_with_sub = 0
for f in sorted(glob.glob(os.path.join(diffdir, "*.json"))):
    o = json.load(open(f))
    pt = o.get("per_turn_diff", []) or []
    if o.get("turn_alignment") != "ok":
        continue
    subturns = [t for t in pt if t.get("n_sub_tito", 0) > 0 or t.get("n_sub_mimo", 0) > 0]
    if not subturns:
        continue
    nfiles_with_sub += 1
    tito_sp = spans(o["tito_tokens"], [t["tito_len"] for t in pt])
    mimo_sp = spans(o["mimo_tokens"], [t["mimo_len"] for t in pt])
    for t in subturns:
        ti = t["turn_index"]; a, b = tito_sp[ti], mimo_sp[ti]
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        reps = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag == "replace"]
        if not reps:
            continue
        print(f"\n=== {os.path.basename(f)[:40]} turn {ti} (sub_t={t['n_sub_tito']} sub_m={t['n_sub_mimo']}) ===")
        for i1, i2, j1, j2 in reps:
            print(f"  TITO {a[i1:i2]} = {tok.decode(a[i1:i2])!r}")
            print(f"  MIMO {b[j1:j2]} = {tok.decode(b[j1:j2])!r}")
        shown += 1
        if shown >= 8:
            break
    if shown >= 8:
        break
print(f"\n(files with >=1 substitution turn: {nfiles_with_sub})")
if not shown:
    print("no substitution turns found")
