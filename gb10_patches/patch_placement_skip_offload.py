import os, glob
F = "/workspace/tito-dressage/slime/slime/ray/placement_group.py"
s = open(F).read()
MARK = "phase-swap skips the placement-time offload"
old = ("    if args.offload_rollout:\n"
       "        ray.get(rollout_manager.offload.remote())\n")
new = ("    import os as _ps_os\n"
       "    # phase-swap skips the placement-time offload: freeing pools before any generate (never-replayed\n"
       "    # CUDA graph) is fragile in the multi-engine Ray env; the loop frees after rollout 0 generates.\n"
       "    if args.offload_rollout and _ps_os.environ.get('PHASE_SWAP_TRAINER') != '1':\n"
       "        ray.get(rollout_manager.offload.remote())\n")
if MARK in s:
    print("placement_group.py ALREADY PATCHED")
elif old not in s:
    print("placement_group.py OFFLOAD BLOCK NOT FOUND -- aborting")
else:
    open(F, "w").write(s.replace(old, new, 1))
    for pc in glob.glob("/workspace/tito-dressage/slime/slime/ray/__pycache__/placement_group*"):
        os.remove(pc)
    print("placement_group.py PATCHED (skip placement offload for phase-swap):", MARK in open(F).read())
