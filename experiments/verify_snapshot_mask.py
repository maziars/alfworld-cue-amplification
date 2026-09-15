#!/usr/bin/env python3
"""Empirically verify what snapshot's loss mask covers: only the current turn's
output, or also prior turns' re-rendered assistant tokens.
Replicates build_segment_alignment's prompt_assistant_token_count per turn."""
import sys, json, gzip, glob, os
sys.path.insert(0, "/workspace/tito-dressage")

# --- find a real multi-turn trajectory (messages) ---
cands = sorted(glob.glob("/workspace/tito-dressage/experiments/*/trajectories/*.jsonl.gz"))
messages = None; tools = None; src = None
for f in cands:
    try:
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            msgs = d.get("messages") or (d.get("trajectory") or {}).get("messages") or d.get("normalized_messages")
            if not msgs:
                # dig one level
                for v in d.values():
                    if isinstance(v, dict) and v.get("messages"):
                        msgs = v["messages"]; d = v; break
            if msgs and sum(1 for m in msgs if m.get("role") == "assistant") >= 3:
                messages = msgs; tools = d.get("tools"); src = f; break
    except Exception as e:
        continue
    if messages: break
if not messages:
    print("NO multi-turn trajectory with messages found; candidates:", cands[:5]); sys.exit(1)
print(f"src={os.path.basename(src)}  n_messages={len(messages)}  roles={[m.get('role') for m in messages][:12]}...")

from transformers import AutoTokenizer
from dressage.proxy.last_step.prompt_assistant_mask import PromptAssistantMaskBuilder
tok = AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
mb = PromptAssistantMaskBuilder(tok, model_mask_type="qwen3_5")

def prompt_asst_count(msgs):
    # exactly what build_segment_alignment uses on normalized_request_messages
    try:
        pid, am = mb.get_prompt_ids_and_assistant_masks(msgs, tools)
        return sum(int(x) for x in am), len(pid)
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}", 0

asst_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
print(f"assistant turns: {len(asst_idx)} (message indices {asst_idx})\n")
print("Per snapshot turn: how many ASSISTANT tokens are in the PROMPT (= prior turns) that the mask marks trainable:")
for k, ai in enumerate(asst_idx):
    prompt_msgs = messages[:ai]           # conversation before this turn's assistant response = the snapshot "prompt"
    if not prompt_msgs:
        print(f"  turn {k+1}: (first turn, empty prior)"); continue
    n_prior_asst, n_prompt = prompt_asst_count(prompt_msgs)
    print(f"  turn {k+1}: prompt_assistant_token_count = {n_prior_asst}   (prompt_len={n_prompt})")
print("\nIf these counts are >0 and grow with turn -> snapshot RE-TRAINS prior turns' assistant tokens each turn.")
print("If they are 0 -> snapshot trains only the current turn's output (your expectation).")
