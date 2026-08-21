import os
T = "/workspace/tito-dressage/slime/train.py"
s = open(T).read()
MARK = "PHASE-SWAP"
if MARK in s:
    print("train.py ALREADY PATCHED"); raise SystemExit

# 1. ensure `import os`
if "\nimport os\n" not in ("\n" + s):
    s = s.replace("import ray\n", "import os\nimport ray\n", 1)

# 2. offload -> offload + await + reload optimizer (before train)
off_anchor = ("        if args.offload_rollout:\n"
              "            ray.get(rollout_manager.offload.remote())\n")
off_new = ("        if args.offload_rollout:\n"
           "            _offload_refs = ray.get(rollout_manager.offload.remote())\n"
           "            # PHASE-SWAP: await the SGLang free, then reload optimizer (mem now available) before train.\n"
           "            if os.environ.get('PHASE_SWAP_TRAINER') == '1':\n"
           "                if _offload_refs:\n"
           "                    ray.get(_offload_refs)\n"
           "                actor_model.phaseswap_reload()\n")

# 3. update_weights + onload_kv (LOOP version, 8-space indent) -> insert evict between
uw_anchor = ("        actor_model.update_weights()\n\n"
             "        if args.offload_rollout:\n"
             "            ray.get(rollout_manager.onload_kv.remote())\n")
uw_new = ("        actor_model.update_weights()\n\n"
          "        # PHASE-SWAP: evict optimizer AFTER broadcast, BEFORE onload_kv rebuilds pools big.\n"
          "        if os.environ.get('PHASE_SWAP_TRAINER') == '1':\n"
          "            actor_model.phaseswap_evict()\n\n"
          "        if args.offload_rollout:\n"
          "            ray.get(rollout_manager.onload_kv.remote())\n")

ok = True
if off_anchor not in s:
    print("OFFLOAD ANCHOR NOT FOUND"); ok = False
if uw_anchor not in s:
    print("UPDATE_WEIGHTS/ONLOAD_KV ANCHOR NOT FOUND"); ok = False
if ok:
    s = s.replace(off_anchor, off_new, 1).replace(uw_anchor, uw_new, 1)
    open(T, "w").write(s)
    import glob
    for pc in glob.glob("/workspace/tito-dressage/slime/__pycache__/train*"):
        os.remove(pc)
    print("train.py PATCHED: reload+evict inserted, import os:", "\nimport os\n" in ("\n"+open(T).read()))
