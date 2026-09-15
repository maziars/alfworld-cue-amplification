import json, glob, datetime
files = glob.glob("/workspace/tito-dressage/experiments/mimo_run_100/mimo_diff/*.json")
recs = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    td = d.get("trajectory_diff", {}) or {}
    recs.append({
        "ts": float(d.get("timestamp", 0) or 0),
        "ins": td.get("n_insert", 0), "del": td.get("n_delete", 0),
        "subt": td.get("n_sub_tito", 0), "subm": td.get("n_sub_mimo", 0),
        "tot": td.get("total_diff", 0), "tito": td.get("tito_len", 0), "mimo": td.get("mimo_len", 0),
        "align": d.get("turn_alignment"),
    })
recs = [r for r in recs if r["ts"] > 0]
recs.sort(key=lambda r: r["ts"])
print(f"loaded {len(recs)} drift files with timestamps")
# cluster by time gap > 300s (training phase between rollouts)
steps, cur, last = [], [], None
for r in recs:
    if last is not None and r["ts"] - last > 300:
        steps.append(cur); cur = []
    cur.append(r); last = r["ts"]
if cur: steps.append(cur)
print(f"clustered into {len(steps)} rollout-groups (>300s gaps)\n")
def hhmm(ts): return datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
print(f"{'grp':>3} {'n':>4} {'start':>11} {'gapBefore(s)':>12} | {'meanDiff':>8} {'insPT':>6} {'delPT':>6} {'subT':>5} {'subM':>5} {'nChanB':>6} {'align_mm':>8} {'meanTitoTok':>11} {'meanMimoTok':>11}")
prev_end = None
for i, st in enumerate(steps):
    n = len(st)
    start, end = st[0]["ts"], st[-1]["ts"]
    gap = (start - prev_end) if prev_end else 0
    prev_end = end
    ins = sum(r["ins"] for r in st); dele = sum(r["del"] for r in st)
    subt = sum(r["subt"] for r in st); subm = sum(r["subm"] for r in st)
    tot = sum(r["tot"] for r in st)
    tito = sum(r["tito"] for r in st); mimo = sum(r["mimo"] for r in st)
    nchanb = sum(1 for r in st if r["subt"] > 0 or r["subm"] > 0)
    align_mm = sum(1 for r in st if r["align"] != "ok")
    # ins/del "per turn" proxy: ins/(tot assistant turns) unknown; report per-traj instead
    print(f"{i:>3} {n:>4} {hhmm(start):>11} {gap:>12.0f} | {tot/max(n,1):>8.1f} {ins/max(n,1):>6.1f} {dele/max(n,1):>6.1f} {subt:>5} {subm:>5} {nchanb:>6} {align_mm:>8} {tito/max(n,1):>11.0f} {mimo/max(n,1):>11.0f}")
