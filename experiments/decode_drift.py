#!/usr/bin/env python3
"""Decode WHICH tokens differ between TITO(sampled) and MIMO(re-render).
Reconstructs per-turn spans from the flat tito_tokens/mimo_tokens + per-turn
lengths, runs difflib, and decodes the differing blocks.
Usage: decode_drift.py <mimo_diff_dir> [tokenizer_path]"""
import json, sys, glob, os, difflib

diffdir = sys.argv[1]
tokpath = sys.argv[2] if len(sys.argv) > 2 else None

from transformers import AutoTokenizer
cands = [tokpath] if tokpath else []
cands += (glob.glob("/workspace/tito-dressage/models/*Qwen3*hf*")
          + glob.glob("/workspace/tito-dressage/models/*Qwen3*[Bb]ase*")
          + glob.glob("/workspace/tito-dressage/models/*Qwen3*Instruct*")
          + glob.glob("/root/.cache/huggingface/hub/*[Qq]wen3*")
          + glob.glob("/workspace/tito-dressage/models/*Qwen3*"))
tok = None
for c in cands:
    if not c or not os.path.exists(c):
        continue
    try:
        tok = AutoTokenizer.from_pretrained(c, trust_remote_code=True)
        print("tokenizer loaded from:", c)
        break
    except Exception as e:
        print("  (skip", c, "->", type(e).__name__, ")")
if tok is None:
    print("NO TOKENIZER; tried:", [c for c in cands if c])
    sys.exit(1)

def spans(flat, lens):
    out, o = [], 0
    for L in lens:
        out.append(flat[o:o + L]); o += L
    return out

files = sorted(glob.glob(os.path.join(diffdir, "*.json")))
shown = 0
for f in files:
    o = json.load(open(f))
    pt = o.get("per_turn_diff", []) or []
    if o.get("turn_alignment") != "ok" or not pt:
        continue
    tito_lens = [t["tito_len"] for t in pt]
    mimo_lens = [t["mimo_len"] for t in pt]
    tito_sp = spans(o["tito_tokens"], tito_lens)
    mimo_sp = spans(o["mimo_tokens"], mimo_lens)
    for ti, t in enumerate(pt):
        if t.get("equal", True):
            continue
        a, b = tito_sp[ti], mimo_sp[ti]
        print(f"\n=== {os.path.basename(f)[:44]} turn {ti} (tito {len(a)} vs mimo {len(b)}) ===")
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            ta, tb = a[i1:i2], b[j1:j2]
            print(f"  [{tag}] TITO ids{ta} = {tok.decode(ta)!r}   ||   MIMO ids{tb} = {tok.decode(tb)!r}")
        # show a little context at the very end of the turn (boundary suspect)
        print(f"  TITO tail ids {a[-4:]} = {tok.decode(a[-4:])!r}")
        print(f"  MIMO tail ids {b[-4:]} = {tok.decode(b[-4:])!r}")
        shown += 1
        break
    if shown >= 4:
        break
if not shown:
    print("no drifting aligned turns found")
