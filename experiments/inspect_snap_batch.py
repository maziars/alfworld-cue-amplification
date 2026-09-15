import torch
from collections import defaultdict
D = "/workspace/tito-dressage/experiments/tito_mito_equiv"
b = torch.load(f"{D}/batches/snapshot.pt", map_location="cpu", weights_only=False)
ss = b["samples"]
byrid = defaultdict(list)
for s in ss:
    byrid[s.get("rollout_id")].append(s)
print(f"total samples={len(ss)}  trajectories(rollout_ids)={len(byrid)}")
# pick the trajectory with the most segments
rid = max(byrid, key=lambda k: len(byrid[k]))
segs = byrid[rid]
print(f"\ntrajectory {rid}: {len(segs)} segments  (per-segment tokens / loss_mask_sum):")
prev = 0
for i, s in enumerate(segs):
    ntok = len(s["tokens"]); nloss = int(sum(s.get("loss_mask") or []))
    print(f"  seg {i:2d}: n_tokens={ntok:6d}  loss_mask_sum={nloss:5d}   (delta vs prev seg loss = {nloss - prev:+d})")
    prev = nloss
tot_tok = sum(len(s["tokens"]) for s in ss)
tot_loss = sum(int(sum(s.get("loss_mask") or [])) for s in ss)
print(f"\nBATCH TOTAL: tokens_forwarded={tot_tok}  loss_bearing_tokens={tot_loss}")
# TITO for comparison
bt = torch.load(f"{D}/batches/tito.pt", map_location="cpu", weights_only=False)
tt = sum(len(s['tokens']) for s in bt['samples']); tl = sum(int(sum(s.get('loss_mask') or [])) for s in bt['samples'])
print(f"TITO  TOTAL: tokens_forwarded={tt}  loss_bearing_tokens={tl}")
print("\nINTERPRETATION:")
print(" - if per-seg loss_mask_sum is ~constant (each ~ one turn's response) => only-current-turn trained (each token once)")
print(" - if it GROWS cumulatively => prior turns re-trained")
print(f" - snapshot loss_bearing == TITO loss_bearing ? {tot_loss == tl}")
