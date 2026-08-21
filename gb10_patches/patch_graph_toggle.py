import os, glob
S = "/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py"
s = open(S).read()
OLD = "server_args.disable_cuda_graph = True  # phaseswap: avoid the cuda-graph destroy->empty_cache"
NEW = "server_args.disable_cuda_graph = (_ps_os2.environ.get('SGLANG_PS_KEEP_GRAPH') != '1')  # phaseswap: KEEP_GRAPH=1 leaves cuda-graph ON"
if "SGLANG_PS_KEEP_GRAPH" in s:
    print("ALREADY TOGGLE-PATCHED")
elif OLD not in s:
    print("OLD disable_cuda_graph LINE NOT FOUND -- aborting (no change)")
else:
    open(S, "w").write(s.replace(OLD, NEW, 1))
    for pc in glob.glob("/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/__pycache__/scheduler*"):
        os.remove(pc)
    print("TOGGLE PATCHED (default still disables; SGLANG_PS_KEEP_GRAPH=1 keeps graphs):", "SGLANG_PS_KEEP_GRAPH" in open(S).read())
