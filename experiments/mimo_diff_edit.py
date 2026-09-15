#!/usr/bin/env python3
"""Add MIMO drift capture to _build_mimo_record: keep BOTH token versions per
trajectory (TITO sampled vs MIMO re-render) + log insertions/deletions/
substitutions/hamming, per-turn and trajectory-level. Additive, env-gated
(DRESSAGE_MIMO_DIFF_LOG), wrapped so it can never break record construction.
Run as root inside container. AST-checked + backed up (.bak_diff)."""
import ast, sys

S = "/workspace/tito-dressage/dressage/proxy/server.py"

ANCHOR = '''        if record_token_versions:
            rv = (
                last_step.response_version
                or (last_step.response_versions[-1] if getattr(last_step, "response_versions", None) else None)
                or getattr(last_step, "request_version", None)
                or "unknown"
            )
            record["full_versions"] = [
                str(rv) if m else _INPUT_TOKEN_VERSION for m in response_mask
            ]
        return record'''

DIFF_BLOCK = '''        if record_token_versions:
            rv = (
                last_step.response_version
                or (last_step.response_versions[-1] if getattr(last_step, "response_versions", None) else None)
                or getattr(last_step, "request_version", None)
                or "unknown"
            )
            record["full_versions"] = [
                str(rv) if m else _INPUT_TOKEN_VERSION for m in response_mask
            ]
        # --- MIMO drift capture (additive, env-gated, never breaks record build) ---
        # Keep BOTH token versions per trajectory: TITO = the SAMPLED response ids
        # (what tito/snapshot trains on) vs MIMO = the assistant tokens of the full
        # chat-template RE-RENDER (what this record trains on). Log per-turn and
        # trajectory diffs: insertions / deletions / substitutions / hamming.
        try:
            import os as _os, difflib as _difflib, json as _json
            _mimo_diff_dir = _os.environ.get("DRESSAGE_MIMO_DIFF_LOG")
            if _mimo_diff_dir:
                def _mimo_diff(a, b):
                    if a == b:
                        return {"equal": True, "tito_len": len(a), "mimo_len": len(b),
                                "n_equal": len(a), "n_insert": 0, "n_delete": 0,
                                "n_sub_tito": 0, "n_sub_mimo": 0,
                                "hamming": 0, "total_diff": 0}
                    _sm = _difflib.SequenceMatcher(a=a, b=b, autojunk=False)
                    n_ins = n_del = n_sa = n_sb = n_eq = 0
                    for tag, i1, i2, j1, j2 in _sm.get_opcodes():
                        if tag == "equal":
                            n_eq += i2 - i1
                        elif tag == "insert":
                            n_ins += j2 - j1
                        elif tag == "delete":
                            n_del += i2 - i1
                        elif tag == "replace":
                            n_sa += i2 - i1
                            n_sb += j2 - j1
                    ham = (sum(1 for x, y in zip(a, b) if x != y)
                           if len(a) == len(b) else None)
                    return {"equal": False, "tito_len": len(a), "mimo_len": len(b),
                            "n_equal": n_eq, "n_insert": n_ins, "n_delete": n_del,
                            "n_sub_tito": n_sa, "n_sub_mimo": n_sb, "hamming": ham,
                            "total_diff": n_ins + n_del + max(n_sa, n_sb)}
                _sampled_turns = [
                    mask_builder._coerce_flat_int_list(s.response_token_ids)
                    for s in session.steps
                    if getattr(s, "response_token_ids", None)
                ]
                _runs, _cur = [], []
                for _t, _m in zip(tokens, response_mask):
                    if _m == 1:
                        _cur.append(_t)
                    elif _cur:
                        _runs.append(_cur)
                        _cur = []
                if _cur:
                    _runs.append(_cur)
                _tito_flat = [t for turn in _sampled_turns for t in turn]
                _mimo_flat = [t for run in _runs for t in run]
                _per_turn = []
                if len(_sampled_turns) == len(_runs):
                    _align = "ok"
                    for _i, (_a, _b) in enumerate(zip(_sampled_turns, _runs)):
                        _d = _mimo_diff(_a, _b)
                        _d["turn_index"] = _i
                        _per_turn.append(_d)
                    _hs = [d["hamming"] for d in _per_turn]
                    _traj = {
                        "tito_len": len(_tito_flat), "mimo_len": len(_mimo_flat),
                        "n_equal": sum(d["n_equal"] for d in _per_turn),
                        "n_insert": sum(d["n_insert"] for d in _per_turn),
                        "n_delete": sum(d["n_delete"] for d in _per_turn),
                        "n_sub_tito": sum(d["n_sub_tito"] for d in _per_turn),
                        "n_sub_mimo": sum(d["n_sub_mimo"] for d in _per_turn),
                        "hamming": (sum(_hs) if all(h is not None for h in _hs) else None),
                        "total_diff": sum(d["total_diff"] for d in _per_turn),
                        "n_turns_with_drift": sum(1 for d in _per_turn if not d["equal"]),
                        "rollup": "per_turn_sum",
                    }
                else:
                    _align = "mismatch:sampled=%d,rerender_runs=%d" % (
                        len(_sampled_turns), len(_runs))
                    _traj = _mimo_diff(_tito_flat, _mimo_flat)
                    _traj["rollup"] = "trajectory_level_fallback"
                _out = {
                    "trajectory_id": trajectory_id,
                    "instance_id": instance_id,
                    "num_steps": len(session.steps),
                    "num_assistant_turns_sampled": len(_sampled_turns),
                    "num_assistant_runs_rerender": len(_runs),
                    "turn_alignment": _align,
                    "any_drift": bool(_traj.get("total_diff", 0)),
                    "trajectory_diff": _traj,
                    "per_turn_diff": _per_turn,
                    "tito_tokens": _tito_flat,
                    "mimo_tokens": _mimo_flat,
                    "timestamp": str(time.time()),
                }
                _os.makedirs(_mimo_diff_dir, exist_ok=True)
                _safe = (str(instance_id).replace("/", "_") + "__"
                         + str(trajectory_id).replace("/", "_"))[:180]
                _fp = _os.path.join(_mimo_diff_dir, _safe + ".json")
                with open(_fp + ".tmp", "w") as _f:
                    _json.dump(_out, _f)
                _os.replace(_fp + ".tmp", _fp)
        except Exception as _e:
            try:
                _d2 = _os.environ.get("DRESSAGE_MIMO_DIFF_LOG")
                if _d2:
                    _os.makedirs(_d2, exist_ok=True)
                    with open(_os.path.join(_d2, "_errors.log"), "a") as _ef:
                        _ef.write("%s: %s: %s\\n" % (
                            trajectory_id, type(_e).__name__, _e))
            except Exception:
                pass
        return record'''

with open(S) as f:
    src = f.read()
n = src.count(ANCHOR)
if n != 1:
    print(f"!! ABORT: anchor count={n} (want 1)")
    sys.exit(2)
if "MIMO drift capture" in src:
    print("!! ABORT: drift block already present")
    sys.exit(3)
new = src.replace(ANCHOR, DIFF_BLOCK)
ast.parse(new)
with open(S + ".bak_diff", "w") as f:
    f.write(src)
with open(S, "w") as f:
    f.write(new)
print("OK server.py: MIMO drift block inserted, AST parses, backup .bak_diff")
