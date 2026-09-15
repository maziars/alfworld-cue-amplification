#!/usr/bin/env python3
"""Scaled MITO-vs-MIMO per-turn structural drift over MANY trajectories.

Same comparison as mimo_drift2 (per-turn render vs whole-conversation render of
each assistant turn), but over all trajectories in the given rollout file(s), and
it REPORTS THE RARE DRIFT EVENTS individually (trajectory, turn, first-diff
position, decoded context) so heavy-tailed drift is visible, not averaged away.
Still render-vs-render (seam class); sampled-vs-retok needs the debug dump.
"""
import gzip, json, sys, glob
from collections import defaultdict

sys.path.insert(0, "/workspace/tito-dressage")
from transformers import AutoTokenizer
from dressage.proxy.last_step.prompt_assistant_mask import PromptAssistantMaskBuilder

_tok = AutoTokenizer.from_pretrained("/workspace/tito-dressage/models/Qwen3.5-4B", trust_remote_code=True)
_mb = PromptAssistantMaskBuilder(_tok, model_mask_type="qwen3_5")


def render_mask(messages, tools=None):
    norm = _mb.normalize_template_messages(messages)
    enc = _mb._apply_chat_template(
        norm, tools=tools, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_assistant_tokens_mask=True,
        chat_template=_mb._mask_chat_template,
    )
    return (_mb._coerce_flat_int_list(enc.get("input_ids", [])),
            _mb._coerce_flat_int_list(enc.get("assistant_masks", [])))


def mask_runs(ids, am):
    runs, cur = [], []
    for t, m in zip(ids, am):
        if m:
            cur.append(t)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs


def load_by_traj(path):
    by = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by[r.get("parent_traj_id") or r.get("session_id") or r.get("rollout_id")].append(r)
    return by


def main():
    files = sys.argv[1:] or sorted(glob.glob(
        "/workspace/tito-dressage/examples/scripts/log/alfworld-qwen3.5-4B-disagg/trajectories/rollout_*.jsonl.gz"))
    n_traj = n_turns = n_drift = n_asst = 0
    turn_count_mismatch = 0
    events = []
    for path in files:
        by = load_by_traj(path)
        for tid, recs in by.items():
            recs = sorted(recs, key=lambda r: int(r.get("segment_index", 0)))
            try:
                fids, fam = render_mask(recs[-1].get("messages") or [])
                mimo = mask_runs(fids, fam)
                mito = []
                for seg in recs:
                    sids, sam = render_mask(seg.get("messages") or [])
                    r = mask_runs(sids, sam)
                    mito.append(r[-1] if r else [])
            except Exception:
                continue
            n_traj += 1
            if len(mito) != len(mimo):
                turn_count_mismatch += 1
            n = min(len(mito), len(mimo))
            for k in range(n):
                a, b = mito[k], mimo[k]
                n_turns += 1; n_asst += len(a)
                if a != b:
                    n_drift += 1
                    fp = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                    ca = _tok.decode(a[max(0, fp - 3):fp + 6])
                    cb = _tok.decode(b[max(0, fp - 3):fp + 6])
                    events.append((str(tid)[:16], k, len(a), len(b), fp, repr(ca), repr(cb)))
    print(f"files={len(files)}  trajectories={n_traj}  turns={n_turns}  assistant_tokens={n_asst}")
    print(f"turn-count-mismatch trajectories (mito#turns != mimo#runs)={turn_count_mismatch}")
    print(f"DRIFTED TURNS = {n_drift} / {n_turns} ({n_drift / max(n_turns, 1):.5%})")
    print(f"DRIFT EVENTS ({len(events)} total, showing up to 80):")
    for ev in events[:80]:
        tid, k, la, lb, fp, da, db = ev
        print(f"  traj {tid} turn {k}  len {la}/{lb}  @tok {fp}:  MITO={da}   MIMO={db}")


if __name__ == "__main__":
    main()
