#!/bin/bash
# ============================================================================
# recover_mimo58.sh (#58 FAITHFUL-TEXT)   (runs ON the dgx1 HOST)
# Faithful-text MIMO: DRESSAGE_MIMO_FAITHFUL_TEXT=1 (trained span = raw sampled
# text re-tokenized; BPE-only drift vs TITO). Otherwise identical to recover_mimo54.sh.
# ----------------------------------------------------------------------------
# GB10 GPU-wedge recovery for the COLOCATE phase-swap run. Updated from
# recover_and_resume_colocate.sh. Sequence:
#   graceful SIGTERM (NEVER -9) -> docker restart BOTH containers (re-attaches
#   the --gpus-all cgroup that the wedge revoked) -> verify a REAL GPU KERNEL
#   runs (not just nvidia-smi) -> clear sglang __pycache__ (pyc-corruption guard)
#   -> rebuild ray head+worker WITH the phase-swap env -> relaunch resuming from
#   the latest saved iter (falls back to the iter-29 base).
#
# WHAT CHANGED vs v1 (recover_and_resume_colocate.sh):
#   1. Ray start on BOTH nodes now EXPORTS the phase-swap env, and the launcher
#      docker-exec carries it too (-e ...). v1 set none -> recovery relaunched
#      WITHOUT phase-swap -> the optimizer stayed resident -> guaranteed OOM.
#   2. Relaunch is the v15/v16 phase-swap config: mem_fraction 0.4 (was 0.35),
#      use_streaming_load=False, sglang_context_len=$SGLANG_CTX, a FRESH run
#      ckpt_save dir (run_v16, not the frombase bank), ckpt_load = latest saved
#      iter under run_v16 else the iter-29 base, exp-name/wandb v16.
#   3. GPU verify (vg1/vg2) now runs a torch matmul kernel -- the wedge can let
#      nvidia-smi pass while every kernel throws "operation not permitted".
#   4. Graceful SIGTERM before the restart + __pycache__ scrub after it; NEVER -9.
#
# Refs: [[gb10-nvml-gpu-access-loss]] (docker restart + ray rebuild + resume),
#       [[pyc-corruption-after-pkill]] (never -9; clear __pycache__),
#       [[gb10-phase-swap-plan]] (launch checklist, ray-start env, ckpts).
# EXIT: 0 recovered+relaunched; 1 GPU would not clear / ray would not rebuild.
# ============================================================================
set +e
DGX2=${DGX2:-10.100.24.2}
DGX1=${DGX1:-10.100.24.1}
SSH2="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ConnectionAttempts=4 $DGX2"
LOG=${SUPERVISOR_LOG:-/home/maziar/nvml_supervisor_colo_v2.log}

# ---- run config (env-overridable; the supervisor exports these) ------------
MODELS=/workspace/tito-dressage/models
EXPDIR=${EXPDIR:-/workspace/tito-dressage/experiments/mimo_run}
SAVE_DEBUG=${SAVE_DEBUG:-"$EXPDIR/debug/{rollout_id}.pt"}
RUN_DIR=${RUN_DIR:-$MODELS/Qwen3.5-4B_colocate_run_v16/}     # ckpt_save (FRESH v16 dir)
BASE_CKPT=${BASE_CKPT:-$MODELS/Qwen3.5-4B_colocate_frombase/} # iter-29 bank (fallback)
EXP_NAME=${EXP_NAME:-phaseswap-run-v16}
WANDB_GROUP=${WANDB_GROUP:-phaseswap-run}
SGLANG_CTX=${SGLANG_CTX:-49152}        # --set sglang_context_len ; may be raised by the user
MEM_FRAC=${MEM_FRAC:-0.4}
NUM_ROLLOUT=${NUM_ROLLOUT:-101}        # MUST stay 101 -> total_iters / LR schedule
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
PHASE_SWAP_OPT_DIR=${PHASE_SWAP_OPT_DIR:-$MODELS/_phaseswap_opt}
RUNLOG=${RUNLOG:-$MODELS/_phaseswap_run_v16.log}
CLEAR_PYCACHE=${CLEAR_PYCACHE:-1}

# phase-swap env, in the two forms we need it
PSENV_EXPORT="SGLANG_PHASESWAP=1 DRESSAGE_COMPACT_FILTER=${DRESSAGE_COMPACT_FILTER:-0} PHASE_SWAP_TRAINER=1 SGLANG_GEN_MAMBA_CACHE_SIZE=auto PHASE_SWAP_OPT_DIR=$PHASE_SWAP_OPT_DIR PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 RAY_DISABLE_USAGE_STATS=1 DRESSAGE_FAITHFUL_TEXT_RESPONSE=${DRESSAGE_FAITHFUL_TEXT_RESPONSE:-1}"
PSENV_E="-e SGLANG_PHASESWAP=1 -e DRESSAGE_COMPACT_FILTER=${DRESSAGE_COMPACT_FILTER:-0} -e PHASE_SWAP_TRAINER=1 -e SGLANG_GEN_MAMBA_CACHE_SIZE=auto -e PHASE_SWAP_OPT_DIR=$PHASE_SWAP_OPT_DIR -e DRESSAGE_FAITHFUL_TEXT_RESPONSE=${DRESSAGE_FAITHFUL_TEXT_RESPONSE:-1}"
CAPENV="-e LOG_DIR=$EXPDIR -e DRESSAGE_LOG_WRITE_MODE=await -e DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR=$EXPDIR/trajectory_payload -e DRESSAGE_TRAJECTORY_JSONL_DIR=$EXPDIR/trajectories -e DRESSAGE_TRAJECTORY_ERROR_LOG_DIR=$EXPDIR/trajectory_err -e DRESSAGE_MIMO_DIFF_LOG=$EXPDIR/mimo_diff -e DRESSAGE_FAITHFUL_TEXT_RESPONSE=${DRESSAGE_FAITHFUL_TEXT_RESPONSE:-1}"

log(){ echo "[recover-colo-v2 $(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- REAL GPU-kernel verify (torch matmul; wedge blocks ALL cuda ops) ------
KTEST='import torch;x=torch.randn(64,64,device="cuda");(x@x).sum().item();print("GPU-OK")'
vg1(){ for i in $(seq 1 12); do docker exec tito python3 -c "$KTEST" 2>/dev/null | grep -q "GPU-OK" && return 0; sleep 5; done; return 1; }
vg2(){ for i in $(seq 1 12); do $SSH2 "docker exec tito python3 -c '$KTEST'" 2>/dev/null | grep -q "GPU-OK" && return 0; sleep 5; done; return 1; }

# ---- ckpt_load = latest saved iter under RUN_DIR, else the iter-29 base -----
resolve_load(){
  local it pad
  it=$(docker exec tito cat "${RUN_DIR}latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc '0-9')
  if [ -n "$it" ] && [ "$it" -gt 0 ] 2>/dev/null; then
    pad=$(printf 'iter_%07d' "$it")
    docker exec tito test -d "${RUN_DIR}${pad}" 2>/dev/null && { echo "$RUN_DIR"; return; }
  fi
  echo "$BASE_CKPT"
}

# ---- graceful teardown (NEVER -9); docker restart clears any straggler ------
graceful_stop(){
  log "graceful SIGTERM of driver/train/sglang (never -9) before restart"
  docker exec tito bash -lc "pkill -TERM -f '[d]ressage.runconfig.launch'; pkill -TERM -f 'python3 -m [t]rain'; true"
  $SSH2 "docker exec tito bash -lc \"pkill -TERM -f '[d]ressage.runconfig.launch'; pkill -TERM -f 'python3 -m [t]rain'; true\""
  sleep 8
  docker exec tito bash -lc "ray stop --force >/dev/null 2>&1; pkill -TERM -f '[s]glang::scheduler'; pkill -TERM -f '[s]glang::detoken'; pkill -TERM -f '[d]ressage.proxy'; true"
  $SSH2 "docker exec tito bash -lc \"ray stop --force >/dev/null 2>&1; pkill -TERM -f '[s]glang::scheduler'; pkill -TERM -f '[s]glang::detoken'; pkill -TERM -f '[d]ressage.proxy'; true\""
  sleep 8
}
clear_pyc(){
  [ "$CLEAR_PYCACHE" = "1" ] || return 0
  log "clearing sglang __pycache__ on both nodes ([[pyc-corruption-after-pkill]] guard)"
  local c='find /usr/local/lib/python3.12/dist-packages/sglang -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true'
  docker exec tito bash -lc "$c"; $SSH2 "docker exec tito bash -lc \"$c\""
}

log "=== COLOCATE-v2 RECOVERY START (run_dir=$RUN_DIR ctx=$SGLANG_CTX) ==="
graceful_stop
log "docker restart tito on dgx1 + dgx2 (re-attaches the GPU cgroup)"
docker restart tito >/dev/null 2>&1
$SSH2 "docker restart tito" >/dev/null 2>&1
sleep 6
vg1 && log "dgx1 GPU-OK (kernel ran)" || { log "FATAL dgx1 GPU still wedged after restart (needs human / CDI / host reboot)"; exit 1; }
vg2 && log "dgx2 GPU-OK (kernel ran)" || { log "FATAL dgx2 GPU still wedged after restart (needs human / CDI / host reboot)"; exit 1; }
clear_pyc
# scrub the phase-swap optimizer spill (scratch; rewritten each run)
docker exec tito bash -lc "rm -rf ${PHASE_SWAP_OPT_DIR}/* 2>/dev/null; true"
$SSH2 "docker exec tito bash -lc \"rm -rf ${PHASE_SWAP_OPT_DIR}/* 2>/dev/null; true\""

log "rebuild ray HEAD on dgx1 (with phase-swap env)"
docker exec tito bash -c "export $PSENV_EXPORT; ray stop --force >/dev/null 2>&1; ray start --head --node-ip-address=$DGX1 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=1 --num-cpus=20 --disable-usage-stats" >/dev/null 2>&1
sleep 10
log "rebuild ray WORKER on dgx2 (with phase-swap env)"
$SSH2 "docker exec tito bash -c 'export $PSENV_EXPORT; ray stop --force >/dev/null 2>&1; ray start --address=$DGX1:6379 --node-ip-address=$DGX2 --num-gpus=1 --num-cpus=20 --disable-usage-stats'" >/dev/null 2>&1
sleep 5
NODES=$(docker exec tito bash -c 'ray status 2>/dev/null | grep -c node_')
log "ray_nodes=$NODES (want 2)"
[ "${NODES:-0}" -ge 2 ] 2>/dev/null || { log "FATAL: ray did not form a 2-node cluster"; exit 1; }

LOAD=$(resolve_load)
if [ "$LOAD" = "$RUN_DIR" ]; then log "RESUME from run_v16 latest ckpt: $LOAD"; else log "START from iter-29 base: $LOAD (run_v16 has no ckpt yet)"; fi
# --- per-node coordinator-marker fix: dgx2 saved ckpts lack .metadata/common.pt/metadata.json+latest.txt ---
if [ "$LOAD" = "$RUN_DIR" ]; then
  _it=$(docker exec tito cat "${RUN_DIR}latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc '0-9')
  if [ -n "$_it" ] && [ "$_it" -gt 0 ] 2>/dev/null; then
    _pad=$(printf 'iter_%07d' "$_it")
    log "sync ckpt markers dgx1->dgx2 for ${_pad}"
    docker exec tito tar -C "$RUN_DIR" -cf - "${_pad}/.metadata" "${_pad}/common.pt" "${_pad}/metadata.json" latest_checkpointed_iteration.txt 2>/dev/null | $SSH2 "docker exec -i tito tar -C $RUN_DIR -xf -" 2>/dev/null && log "markers synced" || log "WARN marker sync failed"
  fi
fi
log "relaunch phase-swap colocate (exp=$EXP_NAME mem_frac=$MEM_FRAC ctx=$SGLANG_CTX save->$RUN_DIR)"
docker exec $PSENV_E $CAPENV -d tito bash -lc "cd /workspace/tito-dressage && python3 -m dressage.runconfig.launch \
  --preset COLOCATE_ALFWORLD_4B_GB10 \
  --set token_build_mode=${TOKEN_BUILD_MODE:-tito} \
  --set save_debug_rollout_data=$SAVE_DEBUG \
  --set rollout_batch_size=${RBS:-2} \
  --set n_samples_per_prompt=${NSAMP:-4} \
  --set global_batch_size=${GBS:-8} \
  --set alfworld_max_steps=${AMS:-30} \
  --set sglang_mem_fraction=$MEM_FRAC \
  --set num_rollout=$NUM_ROLLOUT \
  --set save_interval=$SAVE_INTERVAL \
  --set use_streaming_load=False \
  --set sglang_context_len=$SGLANG_CTX \
  --set ckpt_load=$LOAD \
  --set ckpt_save=$RUN_DIR \
  --set wandb_group=$WANDB_GROUP \
  --exp-name $EXP_NAME \
  > $RUNLOG 2>&1"
sleep 40
log "=== COLOCATE-v2 RECOVERY DONE (relaunched; tail $RUNLOG) ==="
exit 0
