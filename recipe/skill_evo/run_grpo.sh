#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONDA_SH=${CONDA_SH:-"/shared/nas2/yujiz/anaconda3/etc/profile.d/conda.sh"}

set +u
source "${CONDA_SH}"
conda activate verl
set -u

############################################
# 0) GPU pinning (set BEFORE Ray)
############################################
RUN_VARIANT=${RUN_VARIANT:-"base"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES:-7}}
export CUDA_VISIBLE_DEVICES
echo "[INFO] RUN_VARIANT=${RUN_VARIANT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

RAY_ADDRESS=${RAY_ADDRESS:-""}

SANDBOX_FUSION_ROOT=${SANDBOX_FUSION_ROOT:-"/shared/nas2/yujiz/rl/SandboxFusion"}
SANDBOX_SERVICE_ENV=${SANDBOX_SERVICE_ENV:-"sandbox-service"}
SANDBOX_RUNTIME_ENV=${SANDBOX_RUNTIME_ENV:-"sandbox-runtime"}
SANDBOX_HOST=${SANDBOX_HOST:-"127.0.0.1"}
SANDBOX_PORT=${SANDBOX_PORT:-""}
SANDBOX_AUTO_START=${SANDBOX_AUTO_START:-1}
SANDBOX_CONFIG_NAME=${SANDBOX_CONFIG_NAME:-"local"}
SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-""}
SANDBOX_HEALTHCHECK_TIMEOUT_S=${SANDBOX_HEALTHCHECK_TIMEOUT_S:-12}
SANDBOX_START_TIMEOUT_S=${SANDBOX_START_TIMEOUT_S:-90}
SKIP_SANDBOX_HEALTHCHECK=${SKIP_SANDBOX_HEALTHCHECK:-0}

############################################
# 1) Experiment config
############################################
PROJECT_NAME=${PROJECT_NAME:-"deepcoder_grpo"}
EXP_NAME=${EXP_NAME:-"GRPO-DeepCoder-Qwen3-4B-${RUN_VARIANT}"}
MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-4B-Instruct-2507"}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-"grpo"}

EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-${EVAL_EVERY_EPOCHS:-30}}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-30}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT:-true}
BEST_CHECKPOINT_DIRNAME=${BEST_CHECKPOINT_DIRNAME:-"best_reward_checkpoint"}

DEEPCODER_REWARD_MODE=${DEEPCODER_REWARD_MODE:-"primal_dual"}
DEEPCODER_USE_PRIMAL_DUAL=${DEEPCODER_USE_PRIMAL_DUAL:-false}
DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-true}
DEEPCODER_BETA=${DEEPCODER_BETA:-1.0}
DEEPCODER_GAMMA=${DEEPCODER_GAMMA:-1.0}
DEEPCODER_PERF_GATE=${DEEPCODER_PERF_GATE:-0.0}

VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.30}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2}
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-8}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-1}

NNODES=${NNODES:-1}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}

SP_SIZE=${SP_SIZE:-1}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}
OFFLOAD=${OFFLOAD:-true}

GEN_TP=${GEN_TP:-1}

############################################
# 2) Paths
############################################
WORKING_DIR=${WORKING_DIR:-"/shared/nas2/yujiz/rl/verl"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/shared/nas2/yujiz/rl/data"}

CKPTS_ROOT=${CKPTS_ROOT:-"/shared/nas2/yujiz/rl/checkpoints"}
CKPTS_DIR="${CKPTS_ROOT}/${PROJECT_NAME}/${EXP_NAME}"
mkdir -p "${CKPTS_DIR}"

HF_HOME=${HF_HOME:-"${RAY_DATA_HOME}/hf_cache"}
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}"

# DeepCoder Data
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/math/deepcoder_codeforces_train.parquet"}
VAL_FILE=${VAL_FILE:-"${RAY_DATA_HOME}/math/deepcoder_codeforces_val.parquet"}

# Using Tensorboard
TENSORBOARD_DIR="${CKPTS_DIR}/tensorboard"
export TENSORBOARD_DIR
mkdir -p "${TENSORBOARD_DIR}"

############################################
# 3) Derived runtime config
############################################
NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")

TRAIN_NGPUS_PER_NODE=${TRAIN_NGPUS_PER_NODE:-${NUM_GPUS}}
FSDP_SIZE=${FSDP_SIZE:-${TRAIN_NGPUS_PER_NODE}}

if [[ "${TRAIN_PROMPT_MINI_BSZ}" -lt "${FSDP_SIZE}" ]]; then
  TRAIN_PROMPT_MINI_BSZ="${FSDP_SIZE}"
fi
if [[ "${TRAIN_PROMPT_BSZ}" -lt "${TRAIN_PROMPT_MINI_BSZ}" ]]; then
  TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_MINI_BSZ}"
fi
if [[ "${GEN_TP}" -gt "${NUM_GPUS}" ]]; then
  GEN_TP="${NUM_GPUS}"
fi

find_free_port() {
  local base=${1:-6379}
  local limit=${2:-120}
  BASE="${base}" LIMIT="${limit}" python3 - <<'PY'
import os
import socket
import sys

base = int(os.environ["BASE"])
limit = int(os.environ["LIMIT"])

for port in range(base, base + limit + 1):
    sock = socket.socket()
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(port)
    sys.exit(0)

print(base)
PY
}

check_sandbox_fusion() {
  if [[ "${SKIP_SANDBOX_HEALTHCHECK}" == "1" ]]; then
    echo "[WARN] Skipping sandbox health check because SKIP_SANDBOX_HEALTHCHECK=1"
    return 0
  fi

  echo "[INFO] SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL}"
  SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL}" \
  SANDBOX_HEALTHCHECK_TIMEOUT_S="${SANDBOX_HEALTHCHECK_TIMEOUT_S}" \
  python3 - <<'PY'
import json
import os
import sys

import requests

url = os.environ["SANDBOX_FUSION_URL"]
timeout_s = int(os.environ["SANDBOX_HEALTHCHECK_TIMEOUT_S"])
payload = {
    "compile_timeout": 5,
    "run_timeout": 5,
    "code": "print('sandbox_healthcheck_ok')",
    "stdin": "",
    "memory_limit_MB": 128,
    "language": "python",
    "files": {},
    "fetch_files": [],
}
headers = {"Content-Type": "application/json", "Accept": "application/json"}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
except Exception as exc:
    print(f"[ERROR] Sandbox health check request failed for {url}: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] Sandbox health check HTTP {response.status_code} for {url}")
if response.status_code != 200:
    body = response.text.strip().replace("\n", " ")
    if len(body) > 400:
        body = body[:400] + "..."
    print(f"[ERROR] Sandbox health check failed: HTTP {response.status_code}, body={body}", file=sys.stderr)
    sys.exit(1)

try:
    result = response.json()
except Exception as exc:
    body = response.text.strip().replace("\n", " ")
    if len(body) > 400:
        body = body[:400] + "..."
    print(f"[ERROR] Sandbox health check returned non-JSON response: {exc}; body={body}", file=sys.stderr)
    sys.exit(1)

status = result.get("status")
run_result = result.get("run_result") or {}
stdout = (run_result.get("stdout") or "").strip()
stderr = (run_result.get("stderr") or "").strip()

if status != "Success":
    print(
        f"[ERROR] Sandbox health check failed: status={status}, stdout={stdout!r}, stderr={stderr!r}",
        file=sys.stderr,
    )
    sys.exit(1)

if stdout != "sandbox_healthcheck_ok":
    print(
        f"[ERROR] Sandbox health check returned unexpected stdout={stdout!r}, stderr={stderr!r}",
        file=sys.stderr,
    )
    sys.exit(1)

print("[INFO] Sandbox health check passed.")
PY
}

ensure_sandbox_fusion() {
  if check_sandbox_fusion; then
    return 0
  fi

  if [[ "${SANDBOX_AUTO_START}" != "1" ]]; then
    echo "[ERROR] Sandbox health check failed and SANDBOX_AUTO_START=${SANDBOX_AUTO_START}" >&2
    return 1
  fi

  echo "[INFO] Attempting to start local Sandbox Fusion..." >&2

  local sandbox_state_dir="${CKPTS_DIR}/sandbox_fusion"
  mkdir -p "${sandbox_state_dir}"

  SANDBOX_FUSION_URL="$(
    SANDBOX_FUSION_ROOT="${SANDBOX_FUSION_ROOT}" \
    SANDBOX_SERVICE_ENV="${SANDBOX_SERVICE_ENV}" \
    SANDBOX_RUNTIME_ENV="${SANDBOX_RUNTIME_ENV}" \
    SANDBOX_CONFIG_NAME="${SANDBOX_CONFIG_NAME}" \
    SANDBOX_HOST="${SANDBOX_HOST}" \
    SANDBOX_PORT="${SANDBOX_PORT}" \
    SANDBOX_START_TIMEOUT_S="${SANDBOX_START_TIMEOUT_S}" \
    SANDBOX_STATE_DIR="${sandbox_state_dir}" \
    SANDBOX_LOG_PATH="${sandbox_state_dir}/sandbox_fusion.log" \
    SANDBOX_PID_FILE="${sandbox_state_dir}/sandbox_fusion.pid" \
    SANDBOX_URL_FILE="${sandbox_state_dir}/sandbox_fusion.url" \
    "${SCRIPT_DIR}/start_sandbox_fusion.sh"
  )"
  export SANDBOX_FUSION_URL

  check_sandbox_fusion
}

if [[ -z "${SANDBOX_PORT}" || "${SANDBOX_PORT}" == "auto" ]]; then
  SANDBOX_PORT=$(find_free_port 28080 400)
fi
if [[ -z "${SANDBOX_FUSION_URL}" ]]; then
  SANDBOX_FUSION_URL="http://${SANDBOX_HOST}:${SANDBOX_PORT}/run_code"
fi

export SANDBOX_PORT
export SANDBOX_FUSION_URL

ensure_sandbox_fusion

echo "[INFO] CKPTS_DIR=${CKPTS_DIR}"
echo "[INFO] TENSORBOARD_DIR=${TENSORBOARD_DIR}"
if [[ -n "${RAY_ADDRESS}" ]]; then
  echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS}"
  echo "[INFO] Using existing Ray cluster via ray.init(address=...)"
else
  echo "[INFO] RAY_ADDRESS is unset; verl will start a local Ray runtime via ray.init()."
fi
echo "[INFO] DEEPCODER_REWARD_MODE=${DEEPCODER_REWARD_MODE}"
echo "[INFO] SANDBOX_PORT=${SANDBOX_PORT}"
echo "[INFO] SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL}"
echo "[INFO] EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS}"
echo "[INFO] SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS}"

############################################
# 4) Run training
############################################
cd "${WORKING_DIR}"

ACTOR_MAX_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
RAY_INIT_ARGS=()
if [[ -n "${RAY_ADDRESS}" ]]; then
  RAY_INIT_ARGS+=(++ray_kwargs.ray_init.address="${RAY_ADDRESS}")
fi

# Note: Using python3 -m verl.trainer.main_ppo for standard GRPO
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.truncation='left' \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  +data.gen_batch_size="${GEN_PROMPT_BSZ}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.shuffle=true \
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKENS}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${ACTOR_MAX_TOKENS}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ACTOR_MAX_TOKENS}" \
  actor_rollout_ref.model.path="${MODEL_ID}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_PROMPT_MINI_BSZ}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_UTIL}" \
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  reward_model.reward_manager=naive \
  +reward_model.sandbox_fusion.url="${SANDBOX_FUSION_URL}" \
  ++reward_model.reward_kwargs.deepcoder_reward_mode="${DEEPCODER_REWARD_MODE}" \
  ++reward_model.reward_kwargs.deepcoder_use_primal_dual="${DEEPCODER_USE_PRIMAL_DUAL}" \
  ++reward_model.reward_kwargs.enable_thought="${DEEPCODER_ENABLE_THOUGHT}" \
  ++reward_model.reward_kwargs.beta="${DEEPCODER_BETA}" \
  ++reward_model.reward_kwargs.gamma="${DEEPCODER_GAMMA}" \
  ++reward_model.reward_kwargs.perf_gate="${DEEPCODER_PERF_GATE}" \
  "${RAY_INIT_ARGS[@]}" \
  trainer.logger="['console','tensorboard']" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${TRAIN_NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_EVERY_STEPS}" \
  trainer.test_freq="${EVAL_EVERY_STEPS}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.resume_mode="disable" \
  ++trainer.save_best_checkpoint="${SAVE_BEST_CHECKPOINT}" \
  ++trainer.best_checkpoint_dirname="${BEST_CHECKPOINT_DIRNAME}" \
  trainer.default_hdfs_dir=null \
  2>&1 | tee "${CKPTS_DIR}/train.log"
