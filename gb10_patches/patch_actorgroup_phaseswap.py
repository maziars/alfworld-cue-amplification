import glob, os
G = "/workspace/tito-dressage/slime/slime/ray/actor_group.py"
s = open(G).read()
MARK = "def phaseswap_evict"
anchor = '    def update_weights(self):\n        """Broadcast weights'
methods = (
    "    def phaseswap_reload(self):\n"
    "        return ray.get([a.phaseswap_reload_optimizer.remote() for a in self._actor_handlers])\n"
    "\n"
    "    def phaseswap_evict(self):\n"
    "        return ray.get([a.phaseswap_evict_optimizer.remote() for a in self._actor_handlers])\n"
    "\n"
)
if MARK in s:
    print("actor_group.py ALREADY PATCHED")
elif anchor not in s:
    print("actor_group.py ANCHOR NOT FOUND -- aborting")
else:
    open(G, "w").write(s.replace(anchor, methods + anchor, 1))
    for pc in glob.glob("/workspace/tito-dressage/slime/slime/ray/__pycache__/actor_group*"):
        os.remove(pc)
    print("actor_group.py PATCHED:", MARK in open(G).read())
