#!/usr/bin/env python3
"""Root-side edits (run via docker exec, root): finish MIMO wiring in run_config.py
(mimo assert) + add a reliable save_debug_rollout_data field (emits
--save-debug-rollout-data, per-step batch .pt WITH sampled tokens), and the
trajectory_store mimo->lineage case. AST-checked + backed up."""
import ast, sys

RUNCFG = "/workspace/tito-dressage/dressage/runconfig/run_config.py"
STORE = "/workspace/tito-dressage/dressage/proxy/trajectory_store.py"

def edit(path, repls):
    with open(path) as f:
        src = f.read()
    orig = src
    for old, new in repls:
        n = src.count(old)
        if n != 1:
            print(f"!! ABORT {path}: count={n} (want 1) for:\n---\n{old[:160]}\n---")
            sys.exit(2)
        src = src.replace(old, new)
    ast.parse(src)
    with open(path + ".bak_mimo", "w") as f:
        f.write(orig)
    with open(path, "w") as f:
        f.write(src)
    print(f"OK {path}: {len(repls)} edits, AST parses, backup written")

edit(RUNCFG, [
    # mimo assert
    ('assert cfg.token_build_mode in ("tito", "snapshot"), cfg.token_build_mode',
     'assert cfg.token_build_mode in ("tito", "snapshot", "mimo"), cfg.token_build_mode'),
    # new field: save_debug_rollout_data (reliable token log)
    ('    token_build_mode: str = "tito"  # tito | snapshot',
     '    token_build_mode: str = "tito"  # tito | snapshot | mimo\n'
     '    save_debug_rollout_data: str = ""  # if set, emit --save-debug-rollout-data <path> (per-step batch .pt WITH sampled tokens)'),
    # emit the flag in to_extra_train_args (before final return a)
    ('''        a += ["--save-interval", str(self.save_interval),
              "--empty-unused-memory-level", str(self.empty_unused_memory_level)]
        return a''',
     '''        a += ["--save-interval", str(self.save_interval),
              "--empty-unused-memory-level", str(self.empty_unused_memory_level)]
        if self.save_debug_rollout_data:
            a += ["--save-debug-rollout-data", self.save_debug_rollout_data]
        return a'''),
])

edit(STORE, [
    ('''        if "snapshot" in token_build_modes:
            return "timeline"
        return None''',
     '''        if "snapshot" in token_build_modes:
            return "timeline"
        if "mimo" in token_build_modes:
            return "lineage"
        return None'''),
])
print("=== run_config + trajectory_store edits done ===")
