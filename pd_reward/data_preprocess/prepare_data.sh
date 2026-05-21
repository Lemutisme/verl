#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONDA_SH=${CONDA_SH:-"/shared/nas2/yujiz/anaconda3/etc/profile.d/conda.sh"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/shared/nas2/yujiz/rl/data"}

set +u
source "${CONDA_SH}"
conda activate verl
set -u

echo "=========================================="
echo "[INFO] Running Math Data Preparation..."
echo "=========================================="
python3 "${SCRIPT_DIR}/prepare_math_eval.py"

echo ""
echo "=========================================="
echo "[INFO] Running Math Train Data Repairs..."
echo "=========================================="
if [[ -f "${RAY_DATA_HOME}/math/deepscalar_train.parquet" ]]; then
  python3 "${SCRIPT_DIR}/format_math_prompts.py" \
    --input "${RAY_DATA_HOME}/math/deepscalar_train.parquet" \
    --output "${RAY_DATA_HOME}/math/deepscalar_train_formatted.parquet"
else
  echo "[WARN] Skipping DeepScaleR formatting: ${RAY_DATA_HOME}/math/deepscalar_train.parquet not found"
fi

if [[ -f "${RAY_DATA_HOME}/general365/train.parquet" ]]; then
  python3 "${SCRIPT_DIR}/format_math_prompts.py" \
    --input "${RAY_DATA_HOME}/general365/train.parquet" \
    --output "${RAY_DATA_HOME}/general365/train_formatted.parquet" \
    --force-hash
else
  echo "[WARN] Skipping General365 formatting: ${RAY_DATA_HOME}/general365/train.parquet not found"
fi

echo ""
echo "=========================================="
echo "[INFO] Running Code Data Preparation..."
echo "=========================================="
python3 "${SCRIPT_DIR}/prepare_code_eval.py"

echo ""
echo "=========================================="
echo "[INFO] Running DeepCoder Data Repairs..."
echo "=========================================="
if [[ -f "${RAY_DATA_HOME}/math/deepcoder_full_train.parquet" ]]; then
  python3 "${SCRIPT_DIR}/clean_deepcoder_data.py" \
    --input "${RAY_DATA_HOME}/math/deepcoder_full_train.parquet" \
    --output "${RAY_DATA_HOME}/math/deepcoder_full_train_clean.parquet"
else
  echo "[WARN] Skipping DeepCoder train cleaning: ${RAY_DATA_HOME}/math/deepcoder_full_train.parquet not found"
fi

if [[ -f "${RAY_DATA_HOME}/coding/code_eval_master.parquet" ]]; then
  python3 "${SCRIPT_DIR}/clean_deepcoder_data.py" \
    --input "${RAY_DATA_HOME}/coding/code_eval_master.parquet" \
    --output "${RAY_DATA_HOME}/coding/code_eval_master_clean.parquet"
else
  echo "[WARN] Skipping DeepCoder eval cleaning: ${RAY_DATA_HOME}/coding/code_eval_master.parquet not found"
fi

echo ""
echo "[SUCCESS] Data preparation complete!"
