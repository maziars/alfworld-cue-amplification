#!/usr/bin/env python3
"""Turn the copied capture launcher into the MIMO launcher with 3 minimal edits.
Recovery machinery (graceful_stop / ray rebuild / GPU verify / marker sync) is
left byte-identical. Run as root in container (file is bind-mounted)."""
import sys

P = "/workspace/tito-dressage/experiments/recover_mimo.sh"

with open(P) as f:
    src = f.read()
orig = src

def rep(old, new):
    global src
    n = src.count(old)
    if n != 1:
        print(f"!! ABORT: count={n} (want 1) for:\n---\n{old[:180]}\n---")
        sys.exit(2)
    src = src.replace(old, new)

# (a) add EXPDIR + SAVE_DEBUG vars right after MODELS
rep(
    "MODELS=/workspace/tito-dressage/models\nRUN_DIR=",
    "MODELS=/workspace/tito-dressage/models\n"
    "EXPDIR=${EXPDIR:-/workspace/tito-dressage/experiments/mimo_run}\n"
    'SAVE_DEBUG=${SAVE_DEBUG:-"$EXPDIR/debug/{rollout_id}.pt"}\n'
    "RUN_DIR=",
)

# (b) CAPENV -> EXPDIR-based + MIMO diff log
rep(
    'CAPENV="-e LOG_DIR=/workspace/tito-dressage/experiments/mimo_capture '
    '-e DRESSAGE_LOG_WRITE_MODE=await '
    '-e DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR=/workspace/tito-dressage/experiments/mimo_capture/trajectory_payload '
    '-e DRESSAGE_TRAJECTORY_JSONL_DIR=/workspace/tito-dressage/experiments/mimo_capture/trajectories '
    '-e DRESSAGE_TRAJECTORY_ERROR_LOG_DIR=/workspace/tito-dressage/experiments/mimo_capture/trajectory_err"',
    'CAPENV="-e LOG_DIR=$EXPDIR '
    '-e DRESSAGE_LOG_WRITE_MODE=await '
    '-e DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR=$EXPDIR/trajectory_payload '
    '-e DRESSAGE_TRAJECTORY_JSONL_DIR=$EXPDIR/trajectories '
    '-e DRESSAGE_TRAJECTORY_ERROR_LOG_DIR=$EXPDIR/trajectory_err '
    '-e DRESSAGE_MIMO_DIFF_LOG=$EXPDIR/mimo_diff"',
)

# (c) default token mode -> mimo, and add the reliable debug-dump --set line
rep(
    "  --set token_build_mode=${TOKEN_BUILD_MODE:-snapshot} \\",
    "  --set token_build_mode=${TOKEN_BUILD_MODE:-mimo} \\\n"
    "  --set save_debug_rollout_data=$SAVE_DEBUG \\",
)

with open(P + ".bak", "w") as f:
    f.write(orig)
with open(P, "w") as f:
    f.write(src)
# bash syntax gate
import subprocess
r = subprocess.run(["bash", "-n", P], capture_output=True, text=True)
print("bash -n:", "OK" if r.returncode == 0 else "FAIL\n" + r.stderr)
print("=== 3 edits applied; backup .bak ===")
