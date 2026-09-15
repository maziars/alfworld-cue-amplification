#!/usr/bin/env python3
import sys, json, gzip, glob, os
sys.path.insert(0, "/workspace/tito-dressage")
cands = sorted(glob.glob("/workspace/tito-dressage/experiments/*/trajectories/*.jsonl.gz"))
messages = None; tools = None; src = None
for f in cands:
    try:
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            msgs = d.get("messages") or (d.get("trajectory") or {}).get("messages") or d.get("normalized_messages")
            if not msgs:
                for v in d.values():
                    if isinstance(v, dict) and v.get("messages"):
                        msgs = v["messages"]; d = v; break
            if msgs and sum(1 for m in msgs if m.get("role") == "assistant") >= 3:
                messages = msgs; tools = d.get("tools"); src = f; break
    except Exception:
        continue
    if messages: break
print(f"src={os.path.basename(src)} n_messages={len(messages)}")
print("--- message structure ---")
for i, m in enumerate(messages[:9]):
    c = m.get("content")
    print(f"  [{i}] role={m.get('role')} keys={list(m.keys())} content={type(c).__name__}({(len(c) if hasattr(c,'__len__') else c)}) tool_calls={'tool_calls' in m}")

from transformers import AutoTokenizer
from dressage.proxy.last_step.prompt_assistant_mask import PromptAssistantMaskBuilder
tok = AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
mb = PromptAssistantMaskBuilder(tok, model_mask_type="qwen3_5")

def render(msgs, agp):
    enc = mb._apply_chat_template(msgs, tools=tools, add_generation_prompt=agp, tokenize=True,
                                  return_dict=True, return_assistant_tokens_mask=True,
                                  chat_template=mb._mask_chat_template)
    ids = mb._coerce_flat_int_list(enc.get("input_ids", []) if hasattr(enc, "get") else [])
    am = mb._coerce_flat_int_list(enc.get("assistant_masks", []) if hasattr(enc, "get") else [])
    return ids, am

def runs_of_ones(am):
    out, c = [], 0
    for x in am:
        if x == 1: c += 1
        elif c: out.append(c); c = 0
    if c: out.append(c)
    return out

print("\n--- FULL conversation render (add_generation_prompt=False) ---")
try:
    ids, am = render(messages, False)
    print(f"  n_tokens={len(ids)} n_assistant_masked={sum(am)} per-turn assistant-run lengths={runs_of_ones(am)}")
except Exception as e:
    print("  ERR:", repr(e))

print("\n--- snapshot per-turn PROMPT mask (add_generation_prompt=True on conv up to each turn) ---")
asst_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
for k, ai in enumerate(asst_idx):
    prompt_msgs = messages[:ai]
    if not prompt_msgs:
        print(f"  turn {k+1}: (no prior)"); continue
    try:
        ids, am = render(prompt_msgs, True)
        print(f"  turn {k+1}: prompt_assistant_token_count={sum(am)}  prompt_len={len(ids)}  (>0 => trains prior turns)")
    except Exception as e:
        print(f"  turn {k+1}: ERR {repr(e)}")
