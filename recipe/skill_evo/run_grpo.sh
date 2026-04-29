#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_grpo.sh -reward {ori|new|pd} -model {qwen3-4b|qwen3-8b|deepseek7b|custom} [options]

Options:
  -reward, --reward         Reward preset: ori, new, pd
  -model, --model           Model preset: qwen3-4b, qwen3-8b, deepseek-r1-1.5b, deepseek7b, custom
  -mode, --mode             Alias of -model
  -kl, --kl                 KL mode: loss, reward, none
  -kl-coef, --kl-coef       KL coefficient; default 0.001
  -kl-type, --kl-type       KL estimator type; default low_var_kl for loss, kl for reward
  -model-id, --model-id     Override the Hugging Face model id directly
  -name, --name             Optional run name suffix; default is timestamp + pid
  -gpus, --gpus             Override CUDA_VISIBLE_DEVICES for this run
  -h, --help                Show this help message

Examples:
  bash run_grpo.sh -reward ori -model qwen3-4b
  bash run_grpo.sh -reward new -model qwen3-8b -kl none -gpus 2,3 -name ablation_a
  bash run_grpo.sh -reward pd -model deepseek7b -kl reward -kl-coef 0.001
  bash run_grpo.sh -reward pd -model custom -model-id Qwen/Qwen3-30B-A3B-Instruct-2507
EOF
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

sanitize_token() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's#[^a-z0-9._-]+#-#g; s#-+#-#g; s#(^-|-$)##g'
}

REWARD_KIND=${REWARD_KIND:-"pd"}
MODEL_PRESET=${MODEL_PRESET:-${MODEL_MODE:-"qwen3-4b"}}
KL_MODE=${KL_MODE:-"loss"}
RUN_NAME=${RUN_NAME:-""}
CLI_MODEL_ID=""
CLI_CUDA_VISIBLE_DEVICES=""
CLI_KL_COEF=""
CLI_KL_TYPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -reward|--reward)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      REWARD_KIND="$2"
      shift 2
      ;;
    -model|--model|-mode|--mode)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      MODEL_PRESET="$2"
      shift 2
      ;;
    -kl|--kl)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      KL_MODE="$2"
      shift 2
      ;;
    -kl-coef|--kl-coef)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      CLI_KL_COEF="$2"
      shift 2
      ;;
    -kl-type|--kl-type)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      CLI_KL_TYPE="$2"
      shift 2
      ;;
    -model-id|--model-id)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      CLI_MODEL_ID="$2"
      shift 2
      ;;
    -name|--name|--run-name)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      RUN_NAME="$2"
      shift 2
      ;;
    -gpus|--gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      CLI_CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${CLI_MODEL_ID}" ]]; then
  MODEL_ID="${CLI_MODEL_ID}"
  MODEL_PRESET="custom"
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONDA_SH=${CONDA_SH:-"/shared/nas2/yujiz/anaconda3/etc/profile.d/conda.sh"}

set +u
source "${CONDA_SH}"
conda activate verl
set -u

REWARD_KIND=$(lower "${REWARD_KIND}")
MODEL_PRESET=$(lower "${MODEL_PRESET}")
KL_MODE=$(lower "${KL_MODE}")

case "${REWARD_KIND}" in
  ori|original)
    RUN_VARIANT="ori"
    REWARD_LABEL="ori"
    REWARD_DEFAULT_GPU=0
    COMBINE_MODE="none"
    DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-false}
    DEEPCODER_BETA=${DEEPCODER_BETA:-0.0}
    DEEPCODER_GAMMA=${DEEPCODER_GAMMA:-0.0}
    ;;
  new|new_reward)
    RUN_VARIANT="new_reward"
    REWARD_LABEL="new"
    REWARD_DEFAULT_GPU=1
    COMBINE_MODE="multiplier"
    DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-true}
    DEEPCODER_BETA=${DEEPCODER_BETA:-1.0}
    DEEPCODER_GAMMA=${DEEPCODER_GAMMA:-1.0}
    ;;
  pd|primal_dual|pd_reward)
    RUN_VARIANT="pd_reward"
    REWARD_LABEL="pd"
    REWARD_DEFAULT_GPU=2
    COMBINE_MODE="pd"
    DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-true}
    DEEPCODER_BETA=${DEEPCODER_BETA:-1.0}
    DEEPCODER_GAMMA=${DEEPCODER_GAMMA:-1.0}
    ;;
  *)
    echo "Unsupported reward preset: ${REWARD_KIND}" >&2
    usage
    exit 1
    ;;
esac

DEEPCODER_PERF_GATE=${DEEPCODER_PERF_GATE:-0.0}

case "${KL_MODE}" in
  none|off|false|no)
    KL_LABEL="kl0"
    USE_KL_LOSS=${USE_KL_LOSS:-false}
    USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-false}
    KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
    KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}
    KL_PENALTY=${KL_PENALTY:-"kl"}
    KL_CTRL_TYPE=${KL_CTRL_TYPE:-"fixed"}
    KL_CTRL_COEF=${KL_CTRL_COEF:-0.0}
    KL_CTRL_TARGET=${KL_CTRL_TARGET:-0.1}
    KL_CTRL_HORIZON=${KL_CTRL_HORIZON:-10000}
    ;;
  loss|actor|actor_loss)
    KL_LABEL="klloss"
    USE_KL_LOSS=${USE_KL_LOSS:-true}
    USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-false}
    KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
    KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}
    KL_PENALTY=${KL_PENALTY:-"kl"}
    KL_CTRL_TYPE=${KL_CTRL_TYPE:-"fixed"}
    KL_CTRL_COEF=${KL_CTRL_COEF:-0.005}
    KL_CTRL_TARGET=${KL_CTRL_TARGET:-0.1}
    KL_CTRL_HORIZON=${KL_CTRL_HORIZON:-10000}
    ;;
  reward|in_reward|reward_penalty)
    KL_LABEL="klreward"
    USE_KL_LOSS=${USE_KL_LOSS:-false}
    USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-true}
    KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
    KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}
    KL_PENALTY=${KL_PENALTY:-"kl"}
    KL_CTRL_TYPE=${KL_CTRL_TYPE:-"fixed"}
    KL_CTRL_COEF=${KL_CTRL_COEF:-0.005}
    KL_CTRL_TARGET=${KL_CTRL_TARGET:-0.1}
    KL_CTRL_HORIZON=${KL_CTRL_HORIZON:-10000}
    ;;
  *)
    echo "Unsupported KL mode: ${KL_MODE}" >&2
    usage
    exit 1
    ;;
esac

if [[ -n "${CLI_KL_COEF}" ]]; then
  case "${KL_MODE}" in
    loss|actor|actor_loss)
      KL_LOSS_COEF="${CLI_KL_COEF}"
      ;;
    reward|in_reward|reward_penalty)
      KL_CTRL_COEF="${CLI_KL_COEF}"
      ;;
  esac
fi

if [[ -n "${CLI_KL_TYPE}" ]]; then
  case "${KL_MODE}" in
    loss|actor|actor_loss)
      KL_LOSS_TYPE="${CLI_KL_TYPE}"
      ;;
    reward|in_reward|reward_penalty)
      KL_PENALTY="${CLI_KL_TYPE}"
      ;;
  esac
fi

case "${MODEL_PRESET}" in
  qwen3-4b|qwen-4b|4b)
    MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-4B-Instruct-2507"}
    MODEL_LABEL="Qwen3-4B"
    ;;
  qwen3-8b|qwen-8b|8b)
    MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-8B"}
    MODEL_LABEL="Qwen3-8B"
    ;;
  deepseek7b|deepseek-7b|ds7b|7b)
    MODEL_ID=${MODEL_ID:-"deepseek-ai/deepseek-llm-7b-chat"}
    MODEL_LABEL="DeepSeek-7B"
    ;;
  deepseek-r1-distill-qwen-1.5b|deepseek-r1-1.5b|r1-1.5b)
    MODEL_ID=${MODEL_ID:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
    MODEL_LABEL="DeepSeek-R1-Distill-Qwen-1.5B"
    ;;
  custom)
    if [[ -z "${MODEL_ID:-}" ]]; then
      echo "-model custom requires -model-id or MODEL_ID" >&2
      exit 1
    fi
    MODEL_LABEL="$(basename "${MODEL_ID}")"
    ;;
  *)
    echo "Unsupported model preset: ${MODEL_PRESET}" >&2
    usage
    exit 1
    ;;
esac

MODEL_TAG=$(sanitize_token "${MODEL_LABEL}")
RUN_INSTANCE=${RUN_INSTANCE:-"$(date +%Y%m%d_%H%M%S)_pid$$"}
RUN_INSTANCE_TAG=$(sanitize_token "${RUN_INSTANCE}")
RUN_NAME=${RUN_NAME:-""}
RUN_TAG=$(sanitize_token "${RUN_NAME}")

if [[ -n "${CLI_CUDA_VISIBLE_DEVICES}" ]]; then
  DEFAULT_CUDA_VISIBLE_DEVICES="${CLI_CUDA_VISIBLE_DEVICES}"
fi

############################################
# 0) GPU pinning (set BEFORE Ray)
############################################
DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-${REWARD_DEFAULT_GPU}}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}
export CUDA_VISIBLE_DEVICES
export RUN_VARIANT

TRACE=${TRACE:-1}
if [[ "${TRACE}" == "1" ]]; then
  set -x
fi

echo "[INFO] RUN_VARIANT=${RUN_VARIANT}"
echo "[INFO] REWARD_KIND=${REWARD_KIND}"
echo "[INFO] MODEL_PRESET=${MODEL_PRESET}"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] KL_MODE=${KL_MODE}"
echo "[INFO] KL_LABEL=${KL_LABEL}"
echo "[INFO] USE_KL_LOSS=${USE_KL_LOSS}"
echo "[INFO] USE_KL_IN_REWARD=${USE_KL_IN_REWARD}"
if [[ "${USE_KL_LOSS}" == "true" ]]; then
  echo "[INFO] KL_LOSS_COEF=${KL_LOSS_COEF}"
  echo "[INFO] KL_LOSS_TYPE=${KL_LOSS_TYPE}"
fi
if [[ "${USE_KL_IN_REWARD}" == "true" ]]; then
  echo "[INFO] KL_CTRL_TYPE=${KL_CTRL_TYPE}"
  echo "[INFO] KL_CTRL_COEF=${KL_CTRL_COEF}"
  echo "[INFO] KL_PENALTY=${KL_PENALTY}"
  echo "[INFO] KL_CTRL_TARGET=${KL_CTRL_TARGET}"
  echo "[INFO] KL_CTRL_HORIZON=${KL_CTRL_HORIZON}"
fi
echo "[INFO] RUN_NAME=${RUN_NAME}"
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
if [[ -n "${RUN_TAG}" ]]; then
  DEFAULT_EXP_NAME="grpo-${MODEL_TAG}-${REWARD_LABEL}-${KL_LABEL}-${RUN_TAG}-${RUN_INSTANCE_TAG}"
else
  DEFAULT_EXP_NAME="grpo-${MODEL_TAG}-${REWARD_LABEL}-${KL_LABEL}-${RUN_INSTANCE_TAG}"
fi
EXP_NAME=${EXP_NAME:-"${DEFAULT_EXP_NAME}"}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-"grpo"}

EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-5}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT:-true}
BEST_CHECKPOINT_DIRNAME=${BEST_CHECKPOINT_DIRNAME:-"best_reward_checkpoint"}
BEST_CHECKPOINT_METRIC=${BEST_CHECKPOINT_METRIC:-"auto"}

VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.15}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-1}
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-4}
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
ALLOW_EXISTING_EXP_DIR=${ALLOW_EXISTING_EXP_DIR:-0}
if [[ -d "${CKPTS_DIR}" ]] && find "${CKPTS_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  if [[ "${ALLOW_EXISTING_EXP_DIR}" != "1" ]]; then
    echo "[ERROR] Existing experiment directory is not empty: ${CKPTS_DIR}" >&2
    echo "[ERROR] Choose a different -name/EXP_NAME/RUN_INSTANCE, or set ALLOW_EXISTING_EXP_DIR=1 if you really want reuse." >&2
    exit 1
  fi
fi
mkdir -p "${CKPTS_DIR}"

RAY_TMP_ROOT=${RAY_TMP_ROOT:-"/shared/nas2/yujiz/rl/ray_tmp"}
RAY_TMP_TAG=${RAY_TMP_TAG:-"$(date +%m%d%H%M%S)_$$"}
RAY_TMPDIR=${RAY_TMPDIR:-"${RAY_TMP_ROOT}"}
mkdir -p "${RAY_TMPDIR}"
if [[ ${#RAY_TMPDIR} -gt 40 ]]; then
  echo "[WARN] RAY_TMPDIR is ${#RAY_TMPDIR} chars: ${RAY_TMPDIR}" >&2
  echo "[WARN] Keep it short, otherwise Ray may hit the AF_UNIX 107-byte socket path limit." >&2
fi

# Override default tempdir so multiprocessing doesn't fill up /tmp
export TMPDIR="${RAY_TMPDIR}"

HF_HOME=${HF_HOME:-"${RAY_DATA_HOME}/hf_cache"}
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}"

TRAIN_FILE="${RAY_DATA_HOME}/math/deepcoder_full_train.parquet"
VAL_FILE="${RAY_DATA_HOME}/math/deepcoder_full_val.parquet"

TENSORBOARD_DIR="${CKPTS_DIR}/tensorboard"
TRAIN_LOG_PATH="${CKPTS_DIR}/train.log"
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

  export SANDBOX_PID_FILE="${sandbox_state_dir}/sandbox_fusion.pid"

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
    SANDBOX_PID_FILE="${SANDBOX_PID_FILE}" \
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

cleanup_sandbox() {
  if [[ -n "${SANDBOX_PID_FILE:-}" ]] && [[ -f "${SANDBOX_PID_FILE}" ]]; then
    local pid=$(cat "${SANDBOX_PID_FILE}")
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      echo "[INFO] Cleaning up local sandbox_fusion process tree (root=${pid})..." >&2
      # Kill child workers first (uvicorn --workers forks children), then parent
      pkill -9 -P "${pid}" 2>/dev/null || true
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup_sandbox EXIT

echo "[INFO] CKPTS_DIR=${CKPTS_DIR}"
echo "[INFO] TRAIN_LOG_PATH=${TRAIN_LOG_PATH}"
echo "[INFO] TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "[INFO] RAY_TMPDIR=${RAY_TMPDIR}"
if [[ -n "${RAY_ADDRESS}" ]]; then
  echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS}"
  echo "[INFO] Using existing Ray cluster via ray.init(address=...)"
else
  echo "[INFO] RAY_ADDRESS is unset; verl will start a local Ray runtime via ray.init()."
fi
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
else
  RAY_INIT_ARGS+=(++ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}")
fi

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
  data.filter_overlong_prompts=true \
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
  actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN:-4096}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
  reward_model.reward_manager=naive \
  ++custom_reward_function.path="${SCRIPT_DIR}/custom_reward.py" \
  ++custom_reward_function.name="compute_score" \
  +reward_model.sandbox_fusion.url="${SANDBOX_FUSION_URL}" \
  ++reward_model.reward_kwargs.combine_mode="${COMBINE_MODE}" \
  ++reward_model.reward_kwargs.enable_thought="${DEEPCODER_ENABLE_THOUGHT}" \
  ++reward_model.reward_kwargs.weight_thought="${DEEPCODER_BETA}" \
  ++reward_model.reward_kwargs.weight_action="${DEEPCODER_GAMMA}" \
  ++reward_model.reward_kwargs.perf_gate="${DEEPCODER_PERF_GATE}" \
  algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
  algorithm.kl_penalty="${KL_PENALTY}" \
  algorithm.kl_ctrl.type="${KL_CTRL_TYPE}" \
  algorithm.kl_ctrl.kl_coef="${KL_CTRL_COEF}" \
  algorithm.kl_ctrl.target_kl="${KL_CTRL_TARGET}" \
  algorithm.kl_ctrl.horizon="${KL_CTRL_HORIZON}" \
  "${RAY_INIT_ARGS[@]}" \
  trainer.logger="['console','tensorboard']" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${TRAIN_NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  ++trainer.val_before_train=true \
  trainer.save_freq="${SAVE_EVERY_STEPS}" \
  trainer.test_freq="${EVAL_EVERY_STEPS}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  ++trainer.save_best_checkpoint="${SAVE_BEST_CHECKPOINT}" \
  ++trainer.best_checkpoint_dirname="${BEST_CHECKPOINT_DIRNAME}" \
  ++trainer.best_checkpoint_metric="${BEST_CHECKPOINT_METRIC}" \
  trainer.default_hdfs_dir=null \
  trainer.validation_data_dir="${CKPTS_DIR}/val_logs" \
  2>&1 | tee "${TRAIN_LOG_PATH}"
