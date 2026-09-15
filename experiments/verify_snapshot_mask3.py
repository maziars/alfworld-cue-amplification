#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace/tito-dressage")
from transformers import AutoTokenizer
from dressage.proxy.last_step.prompt_assistant_mask import PromptAssistantMaskBuilder
tok = AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
mb = PromptAssistantMaskBuilder(tok, model_mask_type="qwen3_5")

msgs = [
    {"role": "system", "content": "You are a helpful ALFWorld agent."},
    {"role": "user", "content": "Task: put a clean mug on the desk."},
    {"role": "assistant", "content": "I should look on the countertop first. ACTION: go to countertop 1."},
    {"role": "user", "content": "Observation: On the countertop 1 you see a mug 1 and a knife 2."},
    {"role": "assistant", "content": "I will take the mug. ACTION: take mug 1 from countertop 1."},
    {"role": "user", "content": "Observation: You pick up the mug 1 from the countertop 1."},
    {"role": "assistant", "content": "Now put it on the desk. ACTION: move mug 1 to desk 1."},
]

def render(m, agp):
    enc = mb._apply_chat_template(m, tools=None, add_generation_prompt=agp, tokenize=True,
                                  return_dict=True, return_assistant_tokens_mask=True,
                                  chat_template=mb._mask_chat_template)
    ids = mb._coerce_flat_int_list(enc.get("input_ids", []))
    am = mb._coerce_flat_int_list(enc.get("assistant_masks", []))
    return ids, am

def runs(am):
    o, c = [], 0
    for x in am:
        if x == 1: c += 1
        elif c: o.append(c); c = 0
    if c: o.append(c)
    return o

ids, am = render(msgs, False)
print(f"FULL render: n_tokens={len(ids)} assistant_masked={sum(am)} per-turn assistant-run lengths={runs(am)}")
print("\nsnapshot per-turn PROMPT mask (this is build_segment_alignment's prompt_assistant_token_count):")
asst_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
for k, ai in enumerate(asst_idx):
    prm = msgs[:ai]
    ids, am = render(prm, True)
    print(f"  snapshot turn {k+1}: prompt_assistant_token_count = {sum(am)}   (prior-turn assistant tokens marked TRAINABLE)")
print("\n=> the loss mask snapshot builds for turn N = (prior-turn assistant tokens) + (turn-N response).")
