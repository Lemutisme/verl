#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash train_code.sh -reward {ori|new|pdpo|gdpo} -model {qwen3-4b|qwen3-8b|deepseek7b|custom} [options]

Options:
  -reward, --reward         Reward preset: ori, new, pdpo, gdpo
  -model, --model           Model preset: qwen3-4b, qwen3-8b, deepseek-r1-1.5b, deepseek7b, custom
  -mode, --mode             Alias of -model
  -kl, --kl                 KL mode: loss, reward, none
  -kl-coef, --kl-coef       KL coefficient; default 0.001
  -kl-type, --kl-type       KL estimator type; default low_var_kl for loss, kl for reward
  -model-id, --model-id     Override the Hugging Face model id directly
  -name, --name             Optional run name suffix; default is timestamp + pid
  -gpus, --gpus             Override CUDA_VISIBLE_DEVICES for this run
  --save_freq               Set checkpoint saving frequency (e.g. -1 to disable)
  -steps, --steps           Total training steps (default: 1000)

Coding sub-reward env knobs:
  CODING_ENABLE_SUB_REWARDS=true/false
  CODING_ENABLE_<NAME>=true/false and CODING_WEIGHT_<NAME>=float
  Names: CODE_EXTRACTABILITY_REWARD, SYNTAX_VALIDITY_REWARD, UNIT_TEST_PASS_RATE,
         COMPILER_RUNTIME_FEEDBACK, STATIC_ANALYSIS_REWARD, EXECUTED_TOKEN_CREDIT,
         BLOCK_LEVEL_PROCESS_REWARD
  EURUS_TRAIN_FILE/EURUS_VAL_FILE override the default Eurus-2-RL parquet files.
  -h, --help                Show this help message

Examples:
  bash train_code.sh -reward ori -model qwen3-4b
  bash train_code.sh -reward new -model qwen3-8b -kl none -gpus 2,3 -name ablation_a
  bash train_code.sh -reward pdpo -model deepseek7b -kl reward -kl-coef 0.001
  bash train_code.sh -reward gdpo -model custom -model-id Qwen/Qwen3-30B-A3B-Instruct-2507
EOF
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

is_truthy() {
  case "$(lower "$1")" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

bracket_csv() {
  local IFS=,
  printf '[%s]' "$*"
}

sanitize_token() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's#[^a-z0-9._-]+#-#g; s#-+#-#g; s#(^-|-$)##g'
}

REWARD_KIND=${REWARD_KIND:-"pdpo"}
MODEL_PRESET=${MODEL_PRESET:-${MODEL_MODE:-"qwen3-4b"}}
KL_MODE=${KL_MODE:-"loss"}
RUN_NAME=${RUN_NAME:-""}
CLI_MODEL_ID=""
CLI_CUDA_VISIBLE_DEVICES=""
CLI_KL_COEF=""
CLI_KL_TYPE=""
CLI_SAVE_FREQ=""
CLI_TOTAL_STEPS=""

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
    --save_freq)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      CLI_SAVE_FREQ="$2"
      shift 2
      ;;
    -steps|--steps)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      CLI_TOTAL_STEPS="$2"
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
    CODING_ENABLE_SUB_REWARDS=${CODING_ENABLE_SUB_REWARDS:-false}
    ;;
  new|new_reward)
    RUN_VARIANT="new_reward"
    REWARD_LABEL="new"
    REWARD_DEFAULT_GPU=1
    COMBINE_MODE="multiplier"
    CODING_ENABLE_SUB_REWARDS=${CODING_ENABLE_SUB_REWARDS:-true}
    ;;
  pdpo|pdpo_reward)
    RUN_VARIANT="pdpo_reward"
    REWARD_LABEL="pdpo"
    REWARD_DEFAULT_GPU=2
    COMBINE_MODE="pdpo"
    ADV_ESTIMATOR="pdpo"
    CODING_ENABLE_SUB_REWARDS=${CODING_ENABLE_SUB_REWARDS:-true}
    ;;
  gdpo|gdpo_reward)
    RUN_VARIANT="gdpo_reward"
    REWARD_LABEL="gdpo"
    REWARD_DEFAULT_GPU=3
    COMBINE_MODE="gdpo"
    ADV_ESTIMATOR="gdpo"
    CODING_ENABLE_SUB_REWARDS=${CODING_ENABLE_SUB_REWARDS:-true}
    ;;
  *)
    echo "Unsupported reward preset: ${REWARD_KIND}" >&2
    usage
    exit 1
    ;;
esac

CODING_PERF_GATE=${CODING_PERF_GATE:--1.0}
CODING_ENABLE_CODE_EXTRACTABILITY_REWARD=${CODING_ENABLE_CODE_EXTRACTABILITY_REWARD:-true}
CODING_WEIGHT_CODE_EXTRACTABILITY_REWARD=${CODING_WEIGHT_CODE_EXTRACTABILITY_REWARD:-0.15}
CODING_ENABLE_SYNTAX_VALIDITY_REWARD=${CODING_ENABLE_SYNTAX_VALIDITY_REWARD:-true}
CODING_WEIGHT_SYNTAX_VALIDITY_REWARD=${CODING_WEIGHT_SYNTAX_VALIDITY_REWARD:-0.25}
CODING_ENABLE_UNIT_TEST_PASS_RATE=${CODING_ENABLE_UNIT_TEST_PASS_RATE:-false}
CODING_WEIGHT_UNIT_TEST_PASS_RATE=${CODING_WEIGHT_UNIT_TEST_PASS_RATE:-0.0}
CODING_ENABLE_COMPILER_RUNTIME_FEEDBACK=${CODING_ENABLE_COMPILER_RUNTIME_FEEDBACK:-true}
CODING_WEIGHT_COMPILER_RUNTIME_FEEDBACK=${CODING_WEIGHT_COMPILER_RUNTIME_FEEDBACK:-0.30}
CODING_ENABLE_STATIC_ANALYSIS_REWARD=${CODING_ENABLE_STATIC_ANALYSIS_REWARD:-false}
CODING_WEIGHT_STATIC_ANALYSIS_REWARD=${CODING_WEIGHT_STATIC_ANALYSIS_REWARD:-0.0}
CODING_ENABLE_EXECUTED_TOKEN_CREDIT=${CODING_ENABLE_EXECUTED_TOKEN_CREDIT:-false}
CODING_WEIGHT_EXECUTED_TOKEN_CREDIT=${CODING_WEIGHT_EXECUTED_TOKEN_CREDIT:-0.0}
CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD=${CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD:-false}
CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD=${CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD:-0.0}

GDPO_ARGS=()
if [[ "${ADV_ESTIMATOR:-grpo}" == "gdpo" ]]; then
  GDPO_MAIN_WEIGHT=${GDPO_MAIN_WEIGHT:-1.0}
  if [[ -z "${GDPO_REWARD_KEYS:-}" ]]; then
    gdpo_keys=("main_reward")
    gdpo_weights=("${GDPO_MAIN_WEIGHT}")
    if is_truthy "${CODING_ENABLE_CODE_EXTRACTABILITY_REWARD}"; then
      gdpo_keys+=("coding_code_extractability_reward")
      gdpo_weights+=("${CODING_WEIGHT_CODE_EXTRACTABILITY_REWARD}")
    fi
    if is_truthy "${CODING_ENABLE_SYNTAX_VALIDITY_REWARD}"; then
      gdpo_keys+=("coding_syntax_validity_reward")
      gdpo_weights+=("${CODING_WEIGHT_SYNTAX_VALIDITY_REWARD}")
    fi
    if is_truthy "${CODING_ENABLE_UNIT_TEST_PASS_RATE}"; then
      gdpo_keys+=("coding_unit_test_pass_rate")
      gdpo_weights+=("${CODING_WEIGHT_UNIT_TEST_PASS_RATE}")
    fi
    if is_truthy "${CODING_ENABLE_COMPILER_RUNTIME_FEEDBACK}"; then
      gdpo_keys+=("coding_compiler_runtime_feedback")
      gdpo_weights+=("${CODING_WEIGHT_COMPILER_RUNTIME_FEEDBACK}")
    fi
    if is_truthy "${CODING_ENABLE_STATIC_ANALYSIS_REWARD}"; then
      gdpo_keys+=("coding_static_analysis_reward")
      gdpo_weights+=("${CODING_WEIGHT_STATIC_ANALYSIS_REWARD}")
    fi
    if is_truthy "${CODING_ENABLE_EXECUTED_TOKEN_CREDIT}"; then
      gdpo_keys+=("coding_executed_token_credit")
      gdpo_weights+=("${CODING_WEIGHT_EXECUTED_TOKEN_CREDIT}")
    fi
    if is_truthy "${CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD}"; then
      gdpo_keys+=("coding_block_level_process_reward")
      gdpo_weights+=("${CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD}")
    fi
    GDPO_REWARD_KEYS=$(bracket_csv "${gdpo_keys[@]}")
    GDPO_REWARD_WEIGHTS=$(bracket_csv "${gdpo_weights[@]}")
  fi
  GDPO_ARGS+=("++algorithm.gdpo_reward_keys=${GDPO_REWARD_KEYS}")
  if [[ -n "${GDPO_REWARD_WEIGHTS:-}" ]]; then
    GDPO_ARGS+=("++algorithm.gdpo_reward_weights=${GDPO_REWARD_WEIGHTS}")
  fi
fi

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
echo "[INFO] CODING_ENABLE_SUB_REWARDS=${CODING_ENABLE_SUB_REWARDS}"
if [[ "${ADV_ESTIMATOR:-grpo}" == "gdpo" ]]; then
  echo "[INFO] GDPO_REWARD_KEYS=${GDPO_REWARD_KEYS}"
  echo "[INFO] GDPO_REWARD_WEIGHTS=${GDPO_REWARD_WEIGHTS:-<equal>}"
fi
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
# These defaults are required for colocated actor/ref/vLLM on one GPU.
# The failing pattern is: step0 checkpoint finishes, vLLM sleep-mode wakes weights,
# then cuMem mapping fails with "CUDA Error: out of memory at cumem_allocator.cpp:62".
# Disabling vLLM/NCCL cuMem symmetric-memory/custom-allreduce paths keeps that wake-up
# from competing with the model/checkpoint allocations. Keep these together with the
# matching Hydra override below; the environment variable alone is not sufficient.
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-true}"
# Keep vLLM sleep-mode cache management enabled. The trainer releases PyTorch's
# cached blocks before wake-up, so this does not require CPU parameter offload.
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-true}"
echo "[INFO] VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM}"
echo "[INFO] NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE}"
echo "[INFO] VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
echo "[INFO] FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE}"

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
PROJECT_NAME=${PROJECT_NAME:-"eurus_grpo"}
if [[ -n "${RUN_TAG}" ]]; then
  DEFAULT_EXP_NAME="grpo-${MODEL_TAG}-${REWARD_LABEL}-${KL_LABEL}-${RUN_TAG}-${RUN_INSTANCE_TAG}"
else
  DEFAULT_EXP_NAME="grpo-${MODEL_TAG}-${REWARD_LABEL}-${KL_LABEL}-${RUN_INSTANCE_TAG}"
fi
EXP_NAME=${EXP_NAME:-"${DEFAULT_EXP_NAME}"}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-"grpo"}

# PDPO hyperparameters
PDPO_BETA_TIE=${PDPO_BETA_TIE:-0.20}
PDPO_BETA_SAME=${PDPO_BETA_SAME:-0.70}
PDPO_LAMBDA_AUX=${PDPO_LAMBDA_AUX:-0.70}
PDPO_MIN_AUX_STD=${PDPO_MIN_AUX_STD:-1e-6}
PDPO_MIN_MAIN_STD=${PDPO_MIN_MAIN_STD:-1e-6}
PDPO_ANSWER_GATE_MIN=${PDPO_ANSWER_GATE_MIN:-0.5}
PDPO_ANSWER_GATE_CLOSED_SCALE=${PDPO_ANSWER_GATE_CLOSED_SCALE:-0.0}
PDPO_CORRECTNESS_SAFE=${PDPO_CORRECTNESS_SAFE:-true}
PDPO_CORRECTNESS_MARGIN=${PDPO_CORRECTNESS_MARGIN:-1e-3}
PDPO_RELIABILITY_ENABLED=${PDPO_RELIABILITY_ENABLED:-true}
PDPO_RELIABILITY_EMA_ALPHA=${PDPO_RELIABILITY_EMA_ALPHA:-0.05}
PDPO_RELIABILITY_MIN_SCALE=${PDPO_RELIABILITY_MIN_SCALE:-0.0}
PDPO_RELIABILITY_MAX_SCALE=${PDPO_RELIABILITY_MAX_SCALE:-1.0}
PDPO_RELIABILITY_TARGET_MARGIN=${PDPO_RELIABILITY_TARGET_MARGIN:-0.02}
PDPO_RELIABILITY_NEGATIVE_TOLERANCE=${PDPO_RELIABILITY_NEGATIVE_TOLERANCE:-0.02}
PDPO_RELIABILITY_WRONG_HIGH_THRESHOLD=${PDPO_RELIABILITY_WRONG_HIGH_THRESHOLD:-0.30}
PDPO_RELIABILITY_WRONG_HIGH_TARGET=${PDPO_RELIABILITY_WRONG_HIGH_TARGET:-0.20}
PDPO_ETA_S=${PDPO_ETA_S:-0.01}
PDPO_LAMBDA_S_MAX=${PDPO_LAMBDA_S_MAX:-2.0}
PDPO_TAU_S=${PDPO_TAU_S:-1.5}
PDPO_SHARPNESS_EMA_ALPHA=${PDPO_SHARPNESS_EMA_ALPHA:-0.1}

SAVE_EVERY_STEPS=${CLI_SAVE_FREQ:-${SAVE_EVERY_STEPS:--1}}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_STEPS=${CLI_TOTAL_STEPS:-${TOTAL_STEPS:-1000}}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-5}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-5}
SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT:-true}
BEST_CHECKPOINT_DIRNAME=${BEST_CHECKPOINT_DIRNAME:-"best_reward_checkpoint"}
BEST_CHECKPOINT_METRIC=${BEST_CHECKPOINT_METRIC:-"auto"}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

NNODES=${NNODES:-1}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}

SP_SIZE=${SP_SIZE:-1}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-true}

GEN_TP=${GEN_TP:-1}

# Model-size-aware defaults for GPU utilization
# Small models (≤4B): no offload, larger batches, more vLLM cache
# Larger models (7-8B): offload enabled, conservative batches
case "${MODEL_PRESET}" in
  qwen3-4b|qwen-4b|4b|deepseek-r1-distill-qwen-1.5b|deepseek-r1-1.5b|r1-1.5b)
    VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.35}
    VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
    TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-4}
    GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-16}
    N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
    TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-4}
    OFFLOAD=${OFFLOAD:-false}
    ;;
  qwen3-8b|qwen-8b|8b|deepseek7b|deepseek-7b|ds7b|7b)
    VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.35}
    VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
    TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2}
    GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-8}
    N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
    TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-2}
    OFFLOAD=${OFFLOAD:-true}
    ;;
  *)
    # Default / custom: conservative
    VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.35}
    VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
    TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2}
    GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-8}
    N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
    TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-2}
    OFFLOAD=${OFFLOAD:-true}
    ;;
esac

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

RAY_TMP_ROOT=${RAY_TMP_ROOT:-"/tmp/ray_yujiz"}
RAY_TMP_TAG=${RAY_TMP_TAG:-"$(date +%m%d%H%M%S)_$$"}
RAY_TMPDIR=${RAY_TMPDIR:-"${RAY_TMP_ROOT}/${RAY_TMP_TAG}"}
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

EURUS_DEFAULT_TRAIN_FILE="${RAY_DATA_HOME}/eurus/eurus_code_train.parquet"
EURUS_DEFAULT_VAL_FILE="${RAY_DATA_HOME}/eurus/eurus_code_val.parquet"
DEEPCODER_DEFAULT_TRAIN_FILE="${RAY_DATA_HOME}/math/deepcoder_full_train.parquet"
DEEPCODER_DEFAULT_VAL_FILE="${RAY_DATA_HOME}/coding/code_eval_master.parquet"
DEEPCODER_CLEAN_TRAIN_FILE="${RAY_DATA_HOME}/math/deepcoder_full_train_clean.parquet"
DEEPCODER_CLEAN_VAL_FILE="${RAY_DATA_HOME}/coding/code_eval_master_clean.parquet"

if [[ -n "${EURUS_TRAIN_FILE:-}" ]]; then
  TRAIN_FILE="${EURUS_TRAIN_FILE}"
elif [[ -f "${EURUS_DEFAULT_TRAIN_FILE}" ]]; then
  TRAIN_FILE="${EURUS_DEFAULT_TRAIN_FILE}"
elif [[ -n "${DEEPCODER_TRAIN_FILE:-}" ]]; then
  TRAIN_FILE="${DEEPCODER_TRAIN_FILE}"
elif [[ -f "${DEEPCODER_CLEAN_TRAIN_FILE}" ]]; then
  TRAIN_FILE="${DEEPCODER_CLEAN_TRAIN_FILE}"
else
  TRAIN_FILE="${DEEPCODER_DEFAULT_TRAIN_FILE}"
fi

if [[ -n "${EURUS_VAL_FILE:-}" ]]; then
  VAL_FILE="${EURUS_VAL_FILE}"
elif [[ -f "${EURUS_DEFAULT_VAL_FILE}" ]]; then
  VAL_FILE="${EURUS_DEFAULT_VAL_FILE}"
elif [[ -n "${DEEPCODER_VAL_FILE:-}" ]]; then
  VAL_FILE="${DEEPCODER_VAL_FILE}"
elif [[ -f "${DEEPCODER_CLEAN_VAL_FILE}" ]]; then
  VAL_FILE="${DEEPCODER_CLEAN_VAL_FILE}"
else
  VAL_FILE="${DEEPCODER_DEFAULT_VAL_FILE}"
fi

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
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-${NUM_GPUS}}

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
  "${GDPO_ARGS[@]}" \
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
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce="${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
  reward_model.reward_manager=naive \
  ++custom_reward_function.path="${SCRIPT_DIR}/custom_reward.py" \
  ++custom_reward_function.name="compute_score" \
  sandbox_fusion.url="${SANDBOX_FUSION_URL}" \
  ++reward_model.reward_kwargs.combine_mode="${COMBINE_MODE}" \
  ++reward_model.reward_kwargs.perf_gate="${CODING_PERF_GATE}" \
  ++reward_model.reward_kwargs.coding_enable_sub_rewards="${CODING_ENABLE_SUB_REWARDS}" \
  ++reward_model.reward_kwargs.coding_enable_code_extractability_reward="${CODING_ENABLE_CODE_EXTRACTABILITY_REWARD}" \
  ++reward_model.reward_kwargs.coding_weight_code_extractability_reward="${CODING_WEIGHT_CODE_EXTRACTABILITY_REWARD}" \
  ++reward_model.reward_kwargs.coding_enable_syntax_validity_reward="${CODING_ENABLE_SYNTAX_VALIDITY_REWARD}" \
  ++reward_model.reward_kwargs.coding_weight_syntax_validity_reward="${CODING_WEIGHT_SYNTAX_VALIDITY_REWARD}" \
  ++reward_model.reward_kwargs.coding_enable_unit_test_pass_rate="${CODING_ENABLE_UNIT_TEST_PASS_RATE}" \
  ++reward_model.reward_kwargs.coding_weight_unit_test_pass_rate="${CODING_WEIGHT_UNIT_TEST_PASS_RATE}" \
  ++reward_model.reward_kwargs.coding_enable_compiler_runtime_feedback="${CODING_ENABLE_COMPILER_RUNTIME_FEEDBACK}" \
  ++reward_model.reward_kwargs.coding_weight_compiler_runtime_feedback="${CODING_WEIGHT_COMPILER_RUNTIME_FEEDBACK}" \
  ++reward_model.reward_kwargs.coding_enable_static_analysis_reward="${CODING_ENABLE_STATIC_ANALYSIS_REWARD}" \
  ++reward_model.reward_kwargs.coding_weight_static_analysis_reward="${CODING_WEIGHT_STATIC_ANALYSIS_REWARD}" \
  ++reward_model.reward_kwargs.coding_enable_executed_token_credit="${CODING_ENABLE_EXECUTED_TOKEN_CREDIT}" \
  ++reward_model.reward_kwargs.coding_weight_executed_token_credit="${CODING_WEIGHT_EXECUTED_TOKEN_CREDIT}" \
  ++reward_model.reward_kwargs.coding_enable_block_level_process_reward="${CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD}" \
  ++reward_model.reward_kwargs.coding_weight_block_level_process_reward="${CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD}" \
  ++reward_model.reward_kwargs.pdpo_beta_tie="${PDPO_BETA_TIE}" \
  ++reward_model.reward_kwargs.pdpo_beta_same="${PDPO_BETA_SAME}" \
  ++reward_model.reward_kwargs.pdpo_lambda_aux="${PDPO_LAMBDA_AUX}" \
  ++reward_model.reward_kwargs.pdpo_min_aux_std="${PDPO_MIN_AUX_STD}" \
  ++reward_model.reward_kwargs.pdpo_min_main_std="${PDPO_MIN_MAIN_STD}" \
  ++reward_model.reward_kwargs.pdpo_answer_gate_min="${PDPO_ANSWER_GATE_MIN}" \
  ++reward_model.reward_kwargs.pdpo_answer_gate_closed_scale="${PDPO_ANSWER_GATE_CLOSED_SCALE}" \
  ++reward_model.reward_kwargs.pdpo_correctness_safe="${PDPO_CORRECTNESS_SAFE}" \
  ++reward_model.reward_kwargs.pdpo_correctness_margin="${PDPO_CORRECTNESS_MARGIN}" \
  ++reward_model.reward_kwargs.pdpo_reliability_enabled="${PDPO_RELIABILITY_ENABLED}" \
  ++reward_model.reward_kwargs.pdpo_reliability_ema_alpha="${PDPO_RELIABILITY_EMA_ALPHA}" \
  ++reward_model.reward_kwargs.pdpo_reliability_min_scale="${PDPO_RELIABILITY_MIN_SCALE}" \
  ++reward_model.reward_kwargs.pdpo_reliability_max_scale="${PDPO_RELIABILITY_MAX_SCALE}" \
  ++reward_model.reward_kwargs.pdpo_reliability_target_margin="${PDPO_RELIABILITY_TARGET_MARGIN}" \
  ++reward_model.reward_kwargs.pdpo_reliability_negative_tolerance="${PDPO_RELIABILITY_NEGATIVE_TOLERANCE}" \
  ++reward_model.reward_kwargs.pdpo_reliability_wrong_high_threshold="${PDPO_RELIABILITY_WRONG_HIGH_THRESHOLD}" \
  ++reward_model.reward_kwargs.pdpo_reliability_wrong_high_target="${PDPO_RELIABILITY_WRONG_HIGH_TARGET}" \
  ++reward_model.reward_kwargs.pdpo_eta_s="${PDPO_ETA_S}" \
  ++reward_model.reward_kwargs.pdpo_lambda_s_max="${PDPO_LAMBDA_S_MAX}" \
  ++reward_model.reward_kwargs.pdpo_tau_s="${PDPO_TAU_S}" \
  ++reward_model.reward_kwargs.pdpo_sharpness_ema_alpha="${PDPO_SHARPNESS_EMA_ALPHA}" \
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
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  ++trainer.save_best_checkpoint="${SAVE_BEST_CHECKPOINT}" \
  ++trainer.best_checkpoint_dirname="${BEST_CHECKPOINT_DIRNAME}" \
  ++trainer.best_checkpoint_metric="${BEST_CHECKPOINT_METRIC}" \
  trainer.default_hdfs_dir=null \
  trainer.validation_data_dir="${CKPTS_DIR}/val_logs" \
  2>&1 | tee "${TRAIN_LOG_PATH}"
