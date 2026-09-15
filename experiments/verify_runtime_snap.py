#!/usr/bin/env python3
"""Call the RUNTIME build_segment_alignment on a synthetic turn-3 base_step and
see whether its final response_mask is cumulative (prior+current) or turn-only."""
import sys, types
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
req_msgs = msgs[:6]               # request/prompt for turn 3 (before the 3rd assistant)
resp_ids = tok.encode(msgs[6]["content"], add_special_tokens=False)

try:
    prompt_ids, _am = mb.get_prompt_ids_and_assistant_masks(req_msgs, None)
except Exception as e:
    print("get_prompt_ids_and_assistant_masks ERR:", repr(e)); sys.exit(1)

bs = types.SimpleNamespace(
    all_token_ids=list(prompt_ids) + list(resp_ids),
    all_logprobs=[0.0] * (len(prompt_ids) + len(resp_ids)),
    all_logprobs_invalid=False,
    normalized_request_messages=req_msgs,
    prompt_token_ids=list(prompt_ids),
    response_token_ids=list(resp_ids),
)
al = mb.build_segment_alignment(bs, None)
rm = al["response_mask"]
print(f"mask_template_equivalent = {al['mask_template_equivalent']}")
print(f"mask_fallback_reason     = {al['mask_fallback_reason']}")
print(f"prompt_assistant_token_count = {al['prompt_assistant_token_count']}  (prior-turn tokens the mask marks trainable)")
print(f"aligned_response_length      = {al['aligned_response_length']} = sum(response_mask)")
print(f"current turn-3 response len   = {len(resp_ids)}")
print(f"prompt_len = {len(prompt_ids)}  total_len = {len(rm)}")
print()
if al["prompt_assistant_token_count"] and al["prompt_assistant_token_count"] > 0:
    print("=> RUNTIME snapshot mask is CUMULATIVE: it re-trains prior turns (needs prompt logprobs).")
else:
    print("=> RUNTIME snapshot mask is TURN-ONLY: trains only the current turn (matches the experiment; no prompt logprobs needed).")
