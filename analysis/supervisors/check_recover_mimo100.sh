#!/bin/bash
# Cron wedge-supervisor for the 100-step MIMO run. Every 15 min:
#  - recover+resume on trainproc death or a stale run-log (hang)
#  - prune checkpoints (keep last-4 + multiples of 25) to bound disk
#  - record the running raw_reward series to mimo100_trend.log
# Safe: flock (no overlap), 30s re-check before recovery, never touches newest ckpt.
set +e
export HOME=${HOME_DIR}
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

LOG=${HOME_DIR}/mimo100_supervisor.log
LOCK=${HOME_DIR}/.mimo100_sup.lock
RUN_DIR=/workspace/tito-dressage/models/Qwen3.5-4B_colocate_mimo100
RUNLOG_H=${PROJECT_ROOT}/models/_mimo_100.log
TARGET=100
HANG=2700   # 45 min stale-log => hang

log(){ echo "[mimo100-sup $(date '+%F %T')] $*" >> "$LOG"; }
exec 9>"$LOCK"; flock -n 9 || exit 0

ITER=$(docker exec tito cat ${RUN_DIR}/latest_checkpointed_iteration.txt 2>/dev/null | tr -dc '0-9'); ITER=${ITER:-0}
TP=$(pgrep -c -f 'python3 -m train')

prune(){
  local its last4 pad keep
  its=$(docker exec tito bash -lc "ls -d ${RUN_DIR}/iter_* 2>/dev/null | sed 's#.*/iter_0*##' | sort -n")
  [ -z "$its" ] && return
  last4=$(echo "$its" | tail -4)
  for it in $its; do
    keep=0
    echo "$last4" | grep -qx "$it" && keep=1
    [ $((it % 25)) -eq 0 ] && keep=1
    if [ "$keep" -eq 0 ]; then
      pad=$(printf 'iter_%07d' "$it")
      docker exec tito rm -rf "${RUN_DIR}/${pad}" 2>/dev/null && log "pruned $pad"
    fi
  done
}

record_reward(){
  local RW
  RW=$(grep -aoE 'raw_reward.?: ?[0-9.]+' "$RUNLOG_H" 2>/dev/null | grep -oE '[0-9.]+' | tr '\n' ' ')
  echo "[$(date '+%F %T')] iter=$ITER trainproc=$TP  raw_reward_series: $RW" > ${HOME_DIR}/mimo100_trend.log
}

do_recover(){
  log "RECOVER+RESUME (iter=$ITER < $TARGET, reason=$1)"
  EXPDIR=/workspace/tito-dressage/experiments/mimo_run_100 \
  RUN_DIR=/workspace/tito-dressage/models/Qwen3.5-4B_colocate_mimo100/ \
  BASE_CKPT=/workspace/tito-dressage/models/Qwen3.5-4B_torch_dist/ \
  TOKEN_BUILD_MODE=mimo RBS=16 NSAMP=8 GBS=128 AMS=30 \
  NUM_ROLLOUT=101 SAVE_INTERVAL=5 MEM_FRAC=0.7 SGLANG_CTX=262144 DRESSAGE_COMPACT_FILTER=1 \
  EXP_NAME=mimo-100-v18cfg WANDB_GROUP=mimo \
  RUNLOG=/workspace/tito-dressage/models/_mimo_100.log \
  SUPERVISOR_LOG=${HOME_DIR}/mimo100_recover.log \
  bash ${PROJECT_ROOT}/experiments/recover_mimo.sh >> "$LOG" 2>&1
  log "recover_mimo.sh returned"
}

record_reward

if [ "$ITER" -ge "$TARGET" ] 2>/dev/null; then
  log "COMPLETE iter=$ITER (no-op, pruning milestones only)"; prune; exit 0
fi

if [ "${TP:-0}" -eq 0 ] 2>/dev/null; then
  sleep 30; TP=$(pgrep -c -f 'python3 -m train')
  if [ "${TP:-0}" -eq 0 ] 2>/dev/null; then do_recover "trainproc=0"; else log "transient trainproc dip; iter=$ITER"; fi
  exit 0
fi

NOW=$(date +%s); LM=$(stat -c %Y "$RUNLOG_H" 2>/dev/null || echo "$NOW"); AGE=$((NOW-LM))
if [ "$AGE" -gt "$HANG" ] 2>/dev/null; then
  log "HANG suspected (log stale ${AGE}s, trainproc=$TP, iter=$ITER)"; do_recover "log-stale-${AGE}s"; exit 0
fi

log "healthy iter=$ITER trainproc=$TP log_age=${AGE}s"
prune
