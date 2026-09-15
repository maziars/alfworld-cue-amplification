import torch, glob, os
fs=sorted(glob.glob("/workspace/tito-dressage/experiments/faithtext_tito/debug/*.pt"),
          key=lambda p:int(os.path.basename(p).split(".")[0]))
print("debug files:", len(fs), "first/last:", os.path.basename(fs[0]), os.path.basename(fs[-1]))
d=torch.load(fs[-1], map_location="cpu", weights_only=False)
print("top keys:", list(d.keys()))
s=d["samples"][0]
print("sample keys:", sorted(s.keys()))
for k,v in s.items():
    try:
        if hasattr(v,"shape"): print(f"   {k}: tensor {tuple(v.shape)} {v.dtype}")
        elif isinstance(v,(list,tuple)): print(f"   {k}: list len={len(v)} head={v[:4]}")
        else: print(f"   {k}: {type(v).__name__} = {repr(v)[:70]}")
    except Exception as e: print("   ",k,"ERR",e)
