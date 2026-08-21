import os, glob
F = "/workspace/tito-dressage/dressage/runconfig/run_config.py"
s = open(F).read()
MARK = "_ps_ofl"
old = '            a += ["--colocate", "--no-offload-train", "--no-offload-rollout"]\n'
new = (
    '            import os as _ps_os\n'
    '            # phase-swap needs the offload/onload dance (needs_offload) so our sglang overrides free/rebuild\n'
    '            _ps_ofl = "--offload-rollout" if _ps_os.environ.get("PHASE_SWAP_TRAINER") == "1" else "--no-offload-rollout"\n'
    '            a += ["--colocate", "--no-offload-train", _ps_ofl]\n'
)
if MARK in s:
    print("run_config.py ALREADY PATCHED")
elif old not in s:
    print("run_config.py OFFLOAD LINE NOT FOUND -- aborting")
else:
    open(F, "w").write(s.replace(old, new, 1))
    for pc in glob.glob("/workspace/tito-dressage/dressage/runconfig/__pycache__/run_config*"):
        os.remove(pc)
    print("run_config.py PATCHED (conditional --offload-rollout):", MARK in open(F).read())
