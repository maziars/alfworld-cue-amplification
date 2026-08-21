import os, glob
S = "/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py"
s = open(S).read()
MARK = "disable_cuda_graph = True  # phaseswap"
anchor = "        _ps_install()\n"
add = ("        import os as _ps_os2\n"
       "        if _ps_os2.environ.get('SGLANG_PHASESWAP') == '1':\n"
       "            server_args.disable_cuda_graph = True  # phaseswap: avoid the cuda-graph destroy->empty_cache\n"
       "            # cudaErrorInvalidValue during pool free/rebuild in the shared-GPU colocate context.\n")
if MARK in s:
    print("ALREADY PATCHED (disable_cuda_graph)")
elif anchor not in s:
    print("INSTALL ANCHOR NOT FOUND -- is the sched inject applied? aborting")
else:
    open(S, "w").write(s.replace(anchor, anchor + add, 1))
    for pc in glob.glob("/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/__pycache__/scheduler*"):
        os.remove(pc)
    print("PATCHED (disable_cuda_graph for phaseswap):", MARK in open(S).read())
