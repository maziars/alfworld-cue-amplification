import torch, json, glob, os
from collections import Counter
dfs = sorted(glob.glob("/workspace/tito-dressage/experiments/mimo_run_100/debug/*.pt"),
             key=lambda p: int(os.path.basename(p).split('.')[0]) if os.path.basename(p).split('.')[0].split('_')[0].isdigit() else 0)
print("n debug dumps:", len(dfs), "names:", [os.path.basename(x) for x in dfs[:6]], "...", [os.path.basename(x) for x in dfs[-3:]])
df = dfs[len(dfs)//2]
d = torch.load(df, map_location="cpu", weights_only=False)
print("mid dump:", os.path.basename(df), "rollout_id:", d.get("rollout_id"), "n_samples:", len(d.get("samples", [])))
s = d["samples"][0]
print("sample keys:", list(s.keys()))
print("  session_id:", s.get("session_id"), "| instance_id:", s.get("instance_id"))
print("  reward:", s.get("reward"), type(s.get("reward")).__name__)
md = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
print("  metadata keys:", list(md.keys())[:10], "| md.session_id:", md.get("session_id"), "| md.instance_id:", md.get("instance_id"))
rews = []
for x in d["samples"]:
    r = x.get("reward")
    if hasattr(r, "item"): r = r.item()
    if r is not None: rews.append(round(float(r), 3))
print("  reward dist in this dump:", Counter(rews))
jf = glob.glob("/workspace/tito-dressage/experiments/mimo_run_100/mimo_diff/*.json")[0]
jd = json.load(open(jf))
print("drift file: trajectory_id:", jd.get("trajectory_id"), "| instance_id:", jd.get("instance_id"))
# does drift.trajectory_id match any debug session_id in this dump?
sids = set(x.get("session_id") for x in d["samples"])
print("  sample session_ids look like:", list(sids)[:3])
