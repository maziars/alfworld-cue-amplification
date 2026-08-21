import glob, os
S = "/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py"
s = open(S).read()
MARK = "phase-swap injection"
anchor = "    # Generate the logger prefix\n"
block = (
    "    # phase-swap injection (dormant unless SGLANG_PHASESWAP=1)\n"
    "    try:\n"
    "        import sys as _ps_sys\n"
    "        if '/workspace/tito-dressage' not in _ps_sys.path:\n"
    "            _ps_sys.path.insert(0, '/workspace/tito-dressage')\n"
    "        from dressage.training.sglang_phaseswap import install as _ps_install\n"
    "        _ps_install()\n"
    "    except Exception as _ps_e:\n"
    "        import logging as _ps_log; _ps_log.getLogger('phaseswap').warning(f'install failed: {_ps_e}')\n"
)
if MARK in s:
    print("ALREADY PATCHED")
elif anchor not in s:
    print("ANCHOR NOT FOUND -- aborting")
else:
    open(S, "w").write(s.replace(anchor, block + anchor, 1))
    for pc in glob.glob("/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/__pycache__/scheduler*"):
        os.remove(pc)
    print("PATCHED run_scheduler_process + cleared pycache; verify:", MARK in open(S).read())
