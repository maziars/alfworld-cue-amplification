#!/usr/bin/env python3
"""Offline MITO-vs-MIMO token-drift analysis for Dressage snapshot-mode runs.

Runs IN-CONTAINER (imports dressage + the Qwen3.5 tokenizer). Uses the REAL
dressage tokenization code paths so the measured drift is MIMO's, not ours.

v1 = FIDELITY PROBE: reconstruct the MIMO (whole-conversation re-render) tokens
and the mask-template assistant mask from the logged messages, and check
(a) mask_template_equivalent, (b) that reconstructed lengths are sane vs the
logged per-segment lengths. If fidelity holds, we extend to the full per-turn
(MITO) vs whole (MIMO) drift, and later to the exact sampled-token version.
"""
import gzip, json, sys
from collections import defaultdict

sys.path.insert(0, "/workspace/tito-dressage")
from transformers import AutoTokenizer
from dressage.proxy.last_step.prompt_assistant_mask import PromptAssistantMaskBuilder

TOK = "/workspace/tito-dressage/models/Qwen3.5-4B"
_tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
_mb = PromptAssistantMaskBuilder(_tok, model_mask_type="qwen3_5")


def mimo_render(messages, tools):
    """Return (default_ids, mask_ids, assistant_mask, mask_template_equivalent)."""
    norm = _mb.normalize_template_messages(messages)
    ids_default = _mb.tokenize_messages(
        norm, tools, add_generation_prompt=False, template_kwargs=None
    )
    enc = _mb._apply_chat_template(
        norm, tools=tools, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_assistant_tokens_mask=True,
        chat_template=_mb._mask_chat_template,
    )
    ids_mask = _mb._coerce_flat_int_list(enc.get("input_ids", []) if hasattr(enc, "get") else [])
    amask = _mb._coerce_flat_int_list(enc.get("assistant_masks", []) if hasattr(enc, "get") else [])
    equiv = (ids_default == ids_mask)
    return ids_default, ids_mask, amask, equiv


def load_by_traj(path):
    by = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = r.get("parent_traj_id") or r.get("session_id") or r.get("rollout_id")
            by[key].append(r)
    return by


def main():
    path = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    by = load_by_traj(path)
    print(f"file={path}  n_trajectories={len(by)}")
    # detect whether any message carries tool schema; ALFWorld passes tools separately,
    # so probe with tools=None first and report the length gap.
    for i, (tid, recs) in enumerate(by.items()):
        if i >= N:
            break
        recs = sorted(recs, key=lambda r: int(r.get("segment_index", 0)))
        full = recs[-1]
        msgs = full.get("messages") or []
        roles = [m.get("role") for m in msgs]
        try:
            ids_d, ids_m, amask, equiv = mimo_render(msgs, None)
        except Exception as e:
            print(f"--- traj {str(tid)[:24]}: RENDER FAILED: {type(e).__name__}: {e}")
            continue
        n_asst_msgs = sum(1 for m in msgs if m.get("role") == "assistant")
        print(f"--- traj {str(tid)[:24]}  segs={len(recs)}  num_steps={full.get('num_steps')}  n_msgs={len(msgs)}  asst_msgs={n_asst_msgs}")
        print(f"    roles: {roles[:6]}{'...' if len(roles)>6 else ''}")
        print(f"    MIMO default_ids len={len(ids_d)}  mask_ids len={len(ids_m)}  mask_template_equivalent={equiv}")
        print(f"    MIMO assistant(loss) tokens={sum(amask)}  ({sum(amask)}/{len(ids_m)} = {sum(amask)/max(len(ids_m),1):.3f})")
        print(f"    logged(last seg): origin_tokens_len={full.get('origin_tokens_len')} prompt_tokens_len={full.get('prompt_tokens_len')} response_length={full.get('response_length')} truncated={full.get('truncated')}")
        # first-segment sanity (prompt only)
        seg0 = recs[0]
        print(f"    seg0: prompt_tokens_len={seg0.get('prompt_tokens_len')} response_length={seg0.get('response_length')} origin={seg0.get('origin_tokens_len')}")


if __name__ == "__main__":
    main()
