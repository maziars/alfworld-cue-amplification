import os, glob
S = "/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py"
s = open(S).read()
MARK = "enable_memory_saver = False  # phaseswap"
# anchor = the last line of the disable_cuda_graph phaseswap block (inside `if SGLANG_PHASESWAP==1:`)
anchor = "            # cudaErrorInvalidValue during pool free/rebuild in the shared-GPU colocate context.\n"
add = ("            server_args.enable_memory_saver = False  # phaseswap: the torch_memory_saver region RETAINS\n"
       "            # physical LPDDR on resize_(0), so the pool FREE returns ~0 to the OS in colocate (measured\n"
       "            # +0.1 vs +36.6 GiB with it off). We do our own resize_(0)/rebuild in place of the stock\n"
       "            # memory_saver release/resume, so memory_saver is unneeded here.\n")
if MARK in s:
    print("ALREADY PATCHED (enable_memory_saver)")
elif anchor not in s:
    print("ANCHOR NOT FOUND -- is the disable_cuda_graph phaseswap patch applied? aborting, no change")
else:
    open(S, "w").write(s.replace(anchor, anchor + add, 1))
    for pc in glob.glob("/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/__pycache__/scheduler*"):
        os.remove(pc)
    ok = MARK in open(S).read()
    print("PATCHED enable_memory_saver=False:", ok)
