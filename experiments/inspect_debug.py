import torch, sys
d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print("top type:", type(d).__name__)
def peek(s, pfx="  "):
    if isinstance(s, dict):
        print(pfx + "sample keys:", list(s.keys()))
        for k in ("tokens", "loss_masks", "loss_mask", "response_length", "rewards", "raw_reward", "advantages"):
            if k in s:
                v = s[k]
                n = len(v) if hasattr(v, "__len__") else v
                print(pfx + f"  {k}: {type(v).__name__} len/val={n}")
    else:
        print(pfx + "sample:", type(s).__name__, str(s)[:80])
if isinstance(d, dict):
    print("keys:", list(d.keys()))
    for k, v in d.items():
        L = len(v) if hasattr(v, "__len__") else ""
        print(f"  {k}: {type(v).__name__} {L}")
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
            peek(v[0], "    ")
elif isinstance(d, (list, tuple)):
    print("len:", len(d))
    peek(d[0])
    # confirm token lists look like the mimo re-render (large ids present)
    s0 = d[0]
    if isinstance(s0, dict) and "tokens" in s0:
        t = s0["tokens"]
        t = t.tolist() if hasattr(t, "tolist") else list(t)
        print("  tokens[:12]:", t[:12])
        print("  has im_start/think ids (248068 think, 248046 im_end):",
              248068 in t, 248046 in t)
