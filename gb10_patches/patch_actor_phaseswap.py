import glob, os
A = "/workspace/tito-dressage/slime/slime/backends/megatron_utils/actor.py"
s = open(A).read()
MARK = "phaseswap_evict_optimizer"
anchor = "    @timer\n    def update_weights(self) -> None:"
methods = (
    "    def phaseswap_reload_optimizer(self):\n"
    "        # GB10 colocate phase-swap: reload optimizer from disk (after SGLang freed its pools, before train).\n"
    "        if os.environ.get('PHASE_SWAP_TRAINER') != '1':\n"
    "            return\n"
    "        from dressage.training.optimizer_phaseswap import get_trainer_swap\n"
    "        get_trainer_swap().reload(self.optimizer)\n"
    "\n"
    "    def phaseswap_evict_optimizer(self):\n"
    "        # GB10 colocate phase-swap: evict optimizer to disk (after update_weights broadcast, before onload_kv).\n"
    "        if os.environ.get('PHASE_SWAP_TRAINER') != '1':\n"
    "            return\n"
    "        from dressage.training.optimizer_phaseswap import get_trainer_swap\n"
    "        get_trainer_swap().evict(self.optimizer)\n"
    "\n"
)
if MARK in s:
    print("actor.py ALREADY PATCHED")
elif anchor not in s:
    print("actor.py ANCHOR NOT FOUND -- aborting")
else:
    assert "import os" in s, "actor.py missing 'import os'"
    open(A, "w").write(s.replace(anchor, methods + anchor, 1))
    for pc in glob.glob("/workspace/tito-dressage/slime/slime/backends/megatron_utils/__pycache__/actor*"):
        os.remove(pc)
    print("actor.py PATCHED:", MARK in open(A).read())
