#!/usr/bin/env python3
"""MITO-vs-MIMO per-turn token-drift (structural), offline, real dressage tokenizer.

For each trajectory (grouped per-segment in the snapshot-run JSONL):
  MITO turn i = assistant tokens of turn i tokenized PER-TURN  (render segment i's
                messages_snapshot; take the last assistant mask-run).
  MIMO turn i = assistant tokens of turn i tokenized in the WHOLE conversation
                (render the final segment's messages; take the i-th mask-run).
Both are the SAME assistant text, tokenized in different contexts. Any difference
is the re-render drift MIMO would train on. We report, per turn and aggregated:
edit distance (ins/del via LCS), exact-match rate, id-substitution rate on the
equal-length overlap, and where drift concentrates (turn index).
"""
import gzip, json, sys
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
    ids = _mb._coerce_flat_int_list(enc.get("input_ids", []))
    am = _mb._coerce_flat_int_list(enc.get("assistant_masks", []))
    return ids, am


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


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        newp = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            newp[j] = prev[j - 1] + 1 if x == y else (prev[j] if prev[j] >= newp[j - 1] else newp[j - 1])
        prev = newp
    return prev[len(b)]


def diff_turn(mito, mimo):
    L = lcs_len(mito, mimo)
    ins = len(mimo) - L          # tokens in MIMO not matched -> inserted by re-render
    dele = len(mito) - L          # tokens in MITO not matched -> deleted by re-render
    editd = ins + dele
    exact = (mito == mimo)
    # position-wise id substitutions on the equal-length overlap (only meaningful pre-first-indel)
    ov = min(len(mito), len(mimo))
    subs = sum(1 for k in range(ov) if mito[k] != mimo[k])
    return dict(len_mito=len(mito), len_mimo=len(mimo), lcs=L, ins=ins, dele=dele,
                edit=editd, exact=exact, overlap=ov, subs_overlap=subs)


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
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    by = load_by_traj(path)
    print(f"file={path}  n_trajectories={len(by)}  analyzing first {N}")
    agg_turns = 0; agg_drift_turns = 0; agg_edit = 0; agg_asst_tok = 0; agg_sub = 0; agg_ov = 0
    per_index_drift = defaultdict(lambda: [0, 0])  # turn_index -> [drifted, total]
    for i, (tid, recs) in enumerate(by.items()):
        if i >= N:
            break
        recs = sorted(recs, key=lambda r: int(r.get("segment_index", 0)))
        full = recs[-1]
        try:
            fids, fam = render_mask(full.get("messages") or [])
            mimo_runs = mask_runs(fids, fam)
            mito_turns = []
            for seg in recs:
                sids, sam = render_mask(seg.get("messages") or [])
                r = mask_runs(sids, sam)
                mito_turns.append(r[-1] if r else [])
        except Exception as e:
            print(f"  traj {str(tid)[:20]}: FAILED {type(e).__name__}: {e}")
            continue
        n = min(len(mito_turns), len(mimo_runs))
        drift_turns = 0; tot_edit = 0; tot_tok = 0; tot_sub = 0; tot_ov = 0
        worst = None
        for k in range(n):
            d = diff_turn(mito_turns[k], mimo_runs[k])
            tot_tok += d["len_mito"]
            tot_edit += d["edit"]; tot_sub += d["subs_overlap"]; tot_ov += d["overlap"]
            drifted = not d["exact"]
            if drifted:
                drift_turns += 1
            per_index_drift[k][0] += int(drifted); per_index_drift[k][1] += 1
            if drifted and (worst is None or d["edit"] > worst[1]["edit"]):
                worst = (k, d)
        reward = full.get("reward_prefilter", full.get("reward"))
        print(f"--- traj {str(tid)[:20]} turns(mito/mimo)={len(mito_turns)}/{len(mimo_runs)} reward={reward} ---")
        print(f"    drifted turns: {drift_turns}/{n} ({drift_turns/max(n,1):.1%})   asst tokens={tot_tok}"
              f"   total edit ops={tot_edit} ({tot_edit/max(tot_tok,1):.3%} of asst tokens)"
              f"   id-sub on overlap={tot_sub}/{tot_ov} ({tot_sub/max(tot_ov,1):.3%})")
        if worst:
            k, d = worst
            print(f"    worst turn #{k}: len mito/mimo={d['len_mito']}/{d['len_mimo']} ins={d['ins']} del={d['dele']} subs={d['subs_overlap']}")
        agg_turns += n; agg_drift_turns += drift_turns; agg_edit += tot_edit
        agg_asst_tok += tot_tok; agg_sub += tot_sub; agg_ov += tot_ov
    print("\n===== AGGREGATE =====")
    print(f"trajectories={min(N,len(by))}  turns={agg_turns}  drifted turns={agg_drift_turns} ({agg_drift_turns/max(agg_turns,1):.1%})")
    print(f"assistant tokens={agg_asst_tok}  edit ops={agg_edit} ({agg_edit/max(agg_asst_tok,1):.3%} of asst tokens)")
    print(f"id-substitutions on equal-len overlap={agg_sub}/{agg_ov} ({agg_sub/max(agg_ov,1):.3%})")
    # drift rate by turn index (first 30)
    idxs = sorted(per_index_drift)[:30]
    print("drift rate by turn index:", " ".join(f"{k}:{per_index_drift[k][0]}/{per_index_drift[k][1]}" for k in idxs[:15]))


if __name__ == "__main__":
    main()
