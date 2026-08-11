#!/bin/bash

# ALFWorld whitebox DISAGGREGATED SYNC training on Qwen3.5-4B, 2-node GB10.
#
# Disaggregated sync: actor(train) and SGLang(rollout) on SEPARATE GPUs.
#   - Deterministic placement (slime/ray/placement_group.py sort_key, IP-ascending
#     PACK): actor -> dgx1 (10.100.24.1, lower IP), SGLang engine -> dgx2 (10.100.24.2).
#   - SGLang ROUTER + Dressage PROXY + trainer/RolloutManager all run on dgx1 (head).
#     The engine on dgx2 registers to the router over CX-7 and advertises its own
#     Ray node IP (10.100.24.2). Weight-sync NCCL master = actor node (10.100.24.1);
#     dgx2 engine joins that group over CX-7.
#
# ATTACHES TO AN ALREADY-RUNNING RAY CLUSTER. This script does NOT ray start /
# ray stop / pkill python|ray. Run it inside the dgx1 head container.

set -ex

export PYTHONBUFFERED=16

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
# Checkpoints live under /workspace/tito-dressage/models (NOT /root) on both nodes.
BASE_FOLDER="${BASE_FOLDER:-/workspace/tito-dressage/models}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/examples/data}"

if [[ ! -f "${SLIME_ROOT}/scripts/models/${MODEL_CONFIG_NAME:-qwen3.5-4B}.sh" ]]; then
  echo "Cannot find slime model config: ${SLIME_ROOT}/scripts/models/${MODEL_CONFIG_NAME:-qwen3.5-4B}.sh" >&2
  echo "Set REPO_ROOT or SLIME_ROOT to match the current checkout layout." >&2
  exit 1
fi

# --- Cross-node addressing over CX-7 (200GbE ConnectX-7) ---
# hostname -i returns 127.0.0.1 and hostname -I leads with 192.168.4.38 on this
# box, so every host below is hard-pinned to the CX-7 subnet 10.100.24.0/24.
MASTER_ADDR="${MASTER_ADDR:-10.100.24.1}"      # dgx1, actor / weight-sync master
DGX2_ADDR="${DGX2_ADDR:-10.100.24.2}"          # dgx2, SGLang engine
SOCKET_IFNAME="${SOCKET_IFNAME:-enp1s0f0np0}"  # CX-7 iface (up on both nodes)

# --- Disaggregated cluster layout: 1 actor GPU (dgx1) + 1 rollout GPU (dgx2) ---
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}       # 1 GPU per GB10 node (rollout placement math)
RAY_NUM_GPUS_PER_NODE=${RAY_NUM_GPUS_PER_NODE:-1}  # informational; cluster already started
CP_SIZE=${CP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}

source "${SLIME_ROOT}/scripts/models/${MODEL_CONFIG_NAME:-qwen3.5-4B}.sh"

# --- Proxy / router configuration (all on dgx1 head) ---
PROXY_HOST=${PROXY_HOST:-0.0.0.0}
PROXY_PORT=${PROXY_PORT:-8800}
PROXY_PUBLIC_HOST=${PROXY_PUBLIC_HOST:-${MASTER_ADDR}}
DRESSAGE_PROXY_URL=${DRESSAGE_PROXY_URL:-http://${PROXY_PUBLIC_HOST}:${PROXY_PORT}}
SGLANG_ROUTER_HOST=${SGLANG_ROUTER_HOST:-${MASTER_ADDR}}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:-8000}
SGLANG_ROUTER_URL=${SGLANG_ROUTER_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}

# --- Checkpoint configuration ---
HF_CHECKPOINT=${HF_CHECKPOINT:-${BASE_FOLDER}/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-${BASE_FOLDER}/Qwen3.5-4B_torch_dist/}
CKPT_LOAD=${CKPT_LOAD:-${BASE_FOLDER}/Qwen3.5-4B_slime/}
CKPT_SAVE=${CKPT_SAVE:-${BASE_FOLDER}/Qwen3.5-4B_slime/}

# --- Training data ---
PROMPT_DATA=${PROMPT_DATA:-${DATA_ROOT}/alfworld/train.jsonl}

# --- ALFWorld agent loop knobs (scaled down for first validation) ---
ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-15}
ALFWORLD_MAX_EPISODE_STEPS=${ALFWORLD_MAX_EPISODE_STEPS:-15}
ALFWORLD_HISTORY_WINDOW=${ALFWORLD_HISTORY_WINDOW:-10}

# --- Model format configuration (proxy + parser) ---
MODEL_MASK_TYPE=${MODEL_MASK_TYPE:-qwen3_5}
MODEL_TOOL_CALL_TYPE=${MODEL_TOOL_CALL_TYPE:-qwen3_5}
TOOL_CALL_PARSE_BACKEND=${TOOL_CALL_PARSE_BACKEND:-sglang_api}
MODEL_REASONING_TYPE=${MODEL_REASONING_TYPE:-qwen3}
REASONING_PARSE_BACKEND=${REASONING_PARSE_BACKEND:-sglang_api}
TOKEN_BUILD_MODE=${TOKEN_BUILD_MODE:-tito}
TITO_MODEL=${TITO_MODEL:-qwen3_5}
TOKENIZER_PATH=${TOKENIZER_PATH:-${HF_CHECKPOINT}}

# --- CX-7 / RoCE NCCL knobs (env-overridable) ---
# Fallback: if RoCE-verbs NCCL hangs at group init, set NCCL_IB_DISABLE=1 to force
# TCP over enp1s0f0np0 (slower but robust).
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f0}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.8}

LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/log/alfworld-qwen3.5-4B-disagg}
DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR=${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR:-${LOG_DIR}/trajectory_payload}
DRESSAGE_TRAJECTORY_ERROR_LOG_DIR=${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR:-${LOG_DIR}/trajectory_err}
PROXY_LOG_FILE=${PROXY_LOG_FILE:-${LOG_DIR}/dressage-proxy.log}
PROXY_PID_FILE=${PROXY_PID_FILE:-${LOG_DIR}/dressage-proxy.pid}
mkdir -p "${LOG_DIR}" "$(dirname "${PROXY_PID_FILE}")" "${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR}" "${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR}"

for TRAJECTORY_LOG_DIR_VAR in DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR DRESSAGE_TRAJECTORY_ERROR_LOG_DIR; do
  TRAJECTORY_LOG_DIR="${!TRAJECTORY_LOG_DIR_VAR}"
  if [[ -z "${TRAJECTORY_LOG_DIR}" || "${TRAJECTORY_LOG_DIR}" == "/" ]]; then
    echo "Refusing to clear unsafe ${TRAJECTORY_LOG_DIR_VAR}: ${TRAJECTORY_LOG_DIR}" >&2
    exit 1
  fi
  echo "Clearing ${TRAJECTORY_LOG_DIR_VAR}: ${TRAJECTORY_LOG_DIR}"
  find "${TRAJECTORY_LOG_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
done

if [[ "${TOKEN_BUILD_MODE}" != "snapshot" && "${TOKEN_BUILD_MODE}" != "tito" ]]; then
  echo "TOKEN_BUILD_MODE must be snapshot or tito, got: ${TOKEN_BUILD_MODE}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
export DRESSAGE_PROXY_URL
export DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR DRESSAGE_TRAJECTORY_ERROR_LOG_DIR
export ALFWORLD_MAX_STEPS ALFWORLD_MAX_EPISODE_STEPS ALFWORLD_HISTORY_WINDOW

COMM_ARGS=(
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
   --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
)

PROXY_ARGS=(
   --sglang-router-url "${SGLANG_ROUTER_URL}"
   --tokenizer-path "${TOKENIZER_PATH}"
   --host "${PROXY_HOST}"
   --port "${PROXY_PORT}"
   --model-mask-type "${MODEL_MASK_TYPE}"
   --model-tool-call-type "${MODEL_TOOL_CALL_TYPE}"
   --tool-call-parse-backend "${TOOL_CALL_PARSE_BACKEND}"
   --model-reasoning-type "${MODEL_REASONING_TYPE}"
   --reasoning-parse-backend "${REASONING_PARSE_BACKEND}"
   --token-build-mode "${TOKEN_BUILD_MODE}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
   --tito-model "${TITO_MODEL}"
)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${CKPT_LOAD}"
   --save "${CKPT_SAVE}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path dressage.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-config-path "${SCRIPT_DIR}/default/dressage_staleness.yaml"
   --custom-generate-function-path "dressage.recipes.alfworld.agent_whitebox.generate"
   --custom-rm-path dressage.reward.custom_rm.custom_rm
   --data-source-path dressage.rollout.data_source.DressageDataSource
   --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process
   --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data
   --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data


   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle

   --num-rollout "${NUM_ROLLOUT:-5}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
   --balance-data
)

EVAL_ARGS=(
   # Multi-segment whitebox does not support evaluation yet.
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 10.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=()
if [ "${USE_WANDB:-1}" = "1" ]; then
  WANDB_ARGS=(
     --use-wandb
     --wandb-project "${WANDB_PROJECT:-dressage-alfworld-gb10}"
     --wandb-group "${WANDB_GROUP:-alfworld-4B-disagg}"
  )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
   --sglang-attention-backend triton
   --sglang-max-mamba-cache-size "${SGLANG_MAX_MAMBA_CACHE:-160}"
   --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL:-4096}"
   --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-128}"
   --sglang-context-length "${SGLANG_CONTEXT_LEN:-49152}"
   --sglang-router-port "${SGLANG_ROUTER_PORT}"
   --router-policy cache_aware
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
)

# --- Start Dressage proxy (on dgx1 head, binds 0.0.0.0:${PROXY_PORT}) ---
# Clear only a stale Dressage proxy (never touches ray / python / sglang engines).
pkill -9 -f "dressage.proxy.server" 2>/dev/null || true
sleep 1
rm -f "${PROXY_PID_FILE}"

cd "${REPO_ROOT}"
python3 -m dressage.proxy.server "${PROXY_ARGS[@]}" >"${PROXY_LOG_FILE}" 2>&1 &
echo $! > "${PROXY_PID_FILE}"
echo "Started Dressage proxy: pid=$(cat "${PROXY_PID_FILE}") log=${PROXY_LOG_FILE}"

cleanup() {
  if [[ -f "${PROXY_PID_FILE}" ]]; then
    PROXY_PID="$(cat "${PROXY_PID_FILE}")"
    kill "${PROXY_PID}" 2>/dev/null || true
    rm -f "${PROXY_PID_FILE}"
  fi
}
trap cleanup EXIT

for i in $(seq 1 60); do
  if curl -sf "${DRESSAGE_PROXY_URL}/health" >/dev/null 2>&1; then
    echo "Dressage proxy is healthy"
    break
  fi
  if [[ "${i}" -eq 60 ]]; then
    echo "Dressage proxy failed health check; see ${PROXY_LOG_FILE}" >&2
    exit 1
  fi
  sleep 1
done

# --- Submit training job to the ALREADY-RUNNING cluster (no ray start here) ---
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${DGX2_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}"
cd "${SLIME_ROOT}"

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${DGX2_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}",
    "SLIME_HOST_IP": "${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_IB_HCA": "${NCCL_IB_HCA}",
    "NCCL_IB_GID_INDEX": "${NCCL_IB_GID_INDEX}",
    "NCCL_IB_DISABLE": "${NCCL_IB_DISABLE}",
    "NCCL_NET_GDR_LEVEL": "${NCCL_NET_GDR_LEVEL}",
    "NCCL_CUMEM_ENABLE": "${NCCL_CUMEM_ENABLE}",
    "NCCL_NVLS_ENABLE": "${NCCL_NVLS_ENABLE}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/:${REPO_ROOT}:${SLIME_ROOT}",
    "DRESSAGE_PROXY_URL": "${DRESSAGE_PROXY_URL}",
    "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR": "${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR}",
    "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR": "${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR}",
    "DRESSAGE_REWARD_MODULES": "dressage.recipes.alfworld.reward",
    "ALFWORLD_MAX_STEPS": "${ALFWORLD_MAX_STEPS}",
    "ALFWORLD_MAX_EPISODE_STEPS": "${ALFWORLD_MAX_EPISODE_STEPS}",
    "ALFWORLD_HISTORY_WINDOW": "${ALFWORLD_HISTORY_WINDOW}",
    "USE_CHUNKED_LOGPROB": "${USE_CHUNKED_LOGPROB:-0}",
    "CHUNKED_LOGPROB_CHUNK_SIZE": "${CHUNKED_LOGPROB_CHUNK_SIZE:-1024}",
    "CHUNKED_LOGPROB_FP32_MATMUL": "${CHUNKED_LOGPROB_FP32_MATMUL:-0}"
  },
  "worker_process_setup_hook": "dressage.training.chunked_logprob.install"
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m train_async \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${COMM_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   ${EXTRA_TRAIN_ARGS:-}
