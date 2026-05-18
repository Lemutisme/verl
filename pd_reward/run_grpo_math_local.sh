#!/usr/bin/env bash
set -euo pipefail

# Local launcher for CSL-server.
# It keeps the training configuration in run_grpo_math.sh unchanged and only
# overrides machine-specific paths/defaults.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
LOCAL_RUN_ROOT=${LOCAL_RUN_ROOT:-"/home/hao/workspace/runs/pd_reward"}

export CONDA_SH=${CONDA_SH:-"/home/hao/miniconda3/etc/profile.d/conda.sh"}
export WORKING_DIR=${WORKING_DIR:-"${REPO_ROOT}"}
export RAY_DATA_HOME=${RAY_DATA_HOME:-"${LOCAL_RUN_ROOT}/data"}
export CKPTS_ROOT=${CKPTS_ROOT:-"${LOCAL_RUN_ROOT}/checkpoints"}
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-"${LOCAL_RUN_ROOT}/ray_tmp"}
export HF_HOME=${HF_HOME:-"${LOCAL_RUN_ROOT}/hf_cache"}
export DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-"0"}

# Current experiments focus on DeepScalar for math and General365 for reasoning.
# Override with: DATASET=general365 bash run_grpo_math_local.sh ...
export DATASET=${DATASET:-"deepscalar"}
export REWARD_KIND=${REWARD_KIND:-"pd"}
export MODEL_PRESET=${MODEL_PRESET:-"qwen3-4b"}

mkdir -p "${RAY_DATA_HOME}" "${CKPTS_ROOT}" "${RAY_TMP_ROOT}" "${HF_HOME}"

case "${DATASET}" in
  gsm8k)
    EXPECTED_TRAIN="${RAY_DATA_HOME}/gsm8k/train.parquet"
    EXPECTED_VAL="${RAY_DATA_HOME}/gsm8k/test.parquet"
    ;;
  deepscalar)
    EXPECTED_TRAIN="${RAY_DATA_HOME}/math/deepscalar_train.parquet"
    EXPECTED_VAL="${RAY_DATA_HOME}/math/deepscalar_val.parquet"
    ;;
  general365)
    EXPECTED_TRAIN="${RAY_DATA_HOME}/general365/train.parquet"
    EXPECTED_VAL="${RAY_DATA_HOME}/general365/test.parquet"
    ;;
  openr1)
    EXPECTED_TRAIN="${RAY_DATA_HOME}/openr1_math/train.parquet"
    EXPECTED_VAL="${RAY_DATA_HOME}/openr1_math/test.parquet"
    ;;
  *)
    EXPECTED_TRAIN=""
    EXPECTED_VAL=""
    ;;
esac

echo "[LOCAL] WORKING_DIR=${WORKING_DIR}"
echo "[LOCAL] RAY_DATA_HOME=${RAY_DATA_HOME}"
echo "[LOCAL] CKPTS_ROOT=${CKPTS_ROOT}"
echo "[LOCAL] RAY_TMP_ROOT=${RAY_TMP_ROOT}"
echo "[LOCAL] HF_HOME=${HF_HOME}"
echo "[LOCAL] DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES}"
echo "[LOCAL] DATASET=${DATASET}"

if [[ -n "${EXPECTED_TRAIN}" && ! -f "${EXPECTED_TRAIN}" ]]; then
  echo "[LOCAL][WARN] Missing train parquet: ${EXPECTED_TRAIN}" >&2
fi
if [[ -n "${EXPECTED_VAL}" && ! -f "${EXPECTED_VAL}" ]]; then
  echo "[LOCAL][WARN] Missing val parquet: ${EXPECTED_VAL}" >&2
fi

exec bash "${SCRIPT_DIR}/run_grpo_math.sh" "$@"
