#!/usr/bin/env python3
"""Aggregate MIMO drift logs (TITO sampled vs MIMO re-render token diffs).
Usage: python3 mimo_diff_analyze.py <mimo_diff_dir>"""
import sys, json, glob, os

d = sys.argv[1] if len(sys.argv) > 1 else "."
files = [f for f in sorted(glob.glob(os.path.join(d, "*.json")))]
n = len(files)
print(f"dir={d}\nn_trajectory_files={n}")
errlog = os.path.join(d, "_errors.log")
if os.path.exists(errlog):
    ne = sum(1 for _ in open(errlog))
    print(f"drift-block _errors.log lines={ne} (should be 0)")
if not n:
    print("NO DIFF FILES YET")
    sys.exit(0)

agg = dict(traj_with_drift=0, tito_tok=0, mimo_tok=0, n_insert=0, n_delete=0,
           n_sub_tito=0, n_sub_mimo=0, hamming=0, hamming_defined=0,
           turns=0, turns_with_drift=0, align_ok=0, align_mismatch=0)
per_traj = []
for f in files:
    try:
        o = json.load(open(f))
    except Exception as e:
        print("skip", os.path.basename(f), e); continue
    td = o.get("trajectory_diff", {}) or {}
    agg["traj_with_drift"] += int(bool(o.get("any_drift")))
    agg["tito_tok"] += td.get("tito_len", 0)
    agg["mimo_tok"] += td.get("mimo_len", 0)
    agg["n_insert"] += td.get("n_insert", 0)
    agg["n_delete"] += td.get("n_delete", 0)
    agg["n_sub_tito"] += td.get("n_sub_tito", 0)
    agg["n_sub_mimo"] += td.get("n_sub_mimo", 0)
    if td.get("hamming") is not None:
        agg["hamming"] += td["hamming"]; agg["hamming_defined"] += 1
    pt = o.get("per_turn_diff", []) or []
    agg["turns"] += len(pt)
    agg["turns_with_drift"] += sum(1 for t in pt if not t.get("equal", True))
    agg["align_ok" if o.get("turn_alignment") == "ok" else "align_mismatch"] += 1
    per_traj.append((os.path.basename(f), td.get("total_diff", 0),
                     td.get("tito_len", 0), td.get("mimo_len", 0),
                     o.get("turn_alignment")))

print("=== AGGREGATE ===")
for k, v in agg.items():
    print(f"  {k}: {v}")
print(f"  trajectory_drift_rate: {agg['traj_with_drift']/n:.3f}")
print(f"  turn_drift_rate: {agg['turns_with_drift']/max(1,agg['turns']):.5f}")
print(f"  net_token_delta (mimo-tito): {agg['mimo_tok']-agg['tito_tok']}")
print(f"  mean tito_len/traj: {agg['tito_tok']/n:.1f}   mean mimo_len/traj: {agg['mimo_tok']/n:.1f}")
print("=== top-12 trajectories by total_diff ===")
for name, tot, tl, ml, al in sorted(per_traj, key=lambda x: -x[1])[:12]:
    print(f"  total_diff={tot:6d}  tito={tl:6d} mimo={ml:6d} align={al}  {name}")
print("=== example drifting turns (first 8) ===")
shown = 0
for f in files:
    if shown >= 8:
        break
    try:
        o = json.load(open(f))
    except Exception:
        continue
    for t in o.get("per_turn_diff", []) or []:
        if not t.get("equal", True):
            print(f"  {os.path.basename(f)[:40]} turn{t.get('turn_index')}: "
                  f"ins={t.get('n_insert')} del={t.get('n_delete')} "
                  f"sub_t={t.get('n_sub_tito')} sub_m={t.get('n_sub_mimo')} "
                  f"ham={t.get('hamming')} tito_len={t.get('tito_len')} mimo_len={t.get('mimo_len')}")
            shown += 1
            if shown >= 8:
                break
