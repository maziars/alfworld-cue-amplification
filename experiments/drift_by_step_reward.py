import torch, json, glob, os
from collections import defaultdict
DBG = "/workspace/tito-dressage/experiments/mimo_run_100/debug"
DRF = "/workspace/tito-dressage/experiments/mimo_run_100/mimo_diff"

# 1. session_id -> (step, reward) from per-step debug dumps
sess = {}
step_r = defaultdict(lambda: [0, 0])  # step -> [n_reward1, n_total]
for df in sorted(glob.glob(f"{DBG}/*.pt"), key=lambda p: int(os.path.basename(p).split('.')[0])):
    d = torch.load(df, map_location="cpu", weights_only=False)
    rid = d.get("rollout_id")
    for s in d.get("samples", []):
        sid = s.get("session_id")
        r = s.get("reward")
        r = r.item() if hasattr(r, "item") else r
        if sid is None or r is None:
            continue
        sess[sid] = (rid, r)
        step_r[rid][1] += 1
        if r >= 0.5:
            step_r[rid][0] += 1
    del d
print(f"debug dumps loaded; sessions mapped={len(sess)}")

# 2. aggregate drift by (step, reward-class)
agg = defaultdict(lambda: defaultdict(float))
unmatched = 0
for f in glob.glob(f"{DRF}/*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    key = sess.get(d.get("trajectory_id"))
    if key is None:
        unmatched += 1
        continue
    step, rew = key
    rc = 1 if rew >= 0.5 else 0
    td = d.get("trajectory_diff", {}) or {}
    a = agg[(step, rc)]
    a["n"] += 1
    for k, tk in [("ins", "n_insert"), ("del", "n_delete"), ("subt", "n_sub_tito"),
                  ("subm", "n_sub_mimo"), ("tot", "total_diff"), ("tito", "tito_len"), ("mimo", "mimo_len")]:
        a[k] += (td.get(tk, 0) or 0)
    if (td.get("n_sub_tito", 0) or 0) > 0 or (td.get("n_sub_mimo", 0) or 0) > 0:
        a["nchanb"] += 1
print(f"unmatched drift files (run-2 orphans/extras): {unmatched}\n")

# 3. table
steps = sorted(set(k[0] for k in agg))
print(f"{'step':>4} {'stepRew':>7} {'rew':>3} {'n':>4} {'meanTotDiff':>11} {'insPT':>6} {'delPT':>6} {'subTok':>7} {'subPT':>6} {'nChanB':>6} {'titoTok':>8} {'mimoTok':>8}")
for st in steps:
    sr = step_r.get(st, [0, 0])
    frac = sr[0] / sr[1] if sr[1] else 0
    for rc in (1, 0):
        a = agg.get((st, rc))
        if not a:
            print(f"{st:>4} {frac:>7.3f} {rc:>3} {0:>4}")
            continue
        n = a["n"]
        print(f"{st:>4} {frac:>7.3f} {rc:>3} {int(n):>4} {a['tot']/n:>11.1f} {a['ins']/n:>6.1f} {a['del']/n:>6.1f} {int(a['subt']+a['subm']):>7} {(a['subt']+a['subm'])/n:>6.1f} {int(a['nchanb']):>6} {a['tito']/n:>8.0f} {a['mimo']/n:>8.0f}")
