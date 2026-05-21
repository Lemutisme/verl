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

if [[ -f "${RAY_DATA_HOME}/general365/test.parquet" ]]; then
  python3 "${SCRIPT_DIR}/format_math_prompts.py" \
    --input "${RAY_DATA_HOME}/general365/test.parquet" \
    --output "${RAY_DATA_HOME}/general365/test_formatted.parquet" \
    --force-hash
else
  echo "[WARN] Skipping General365 eval formatting: ${RAY_DATA_HOME}/general365/test.parquet not found"
fi

echo ""
echo "=========================================="
echo "[INFO] Running DeepScaleR Eval Data Preparation..."
echo "=========================================="
python3 "${SCRIPT_DIR}/prepare_deepscalar_eval.py" \
  --base-eval "${RAY_DATA_HOME}/math/math_eval_master.parquet" \
  --general365-eval "${RAY_DATA_HOME}/general365/test_formatted.parquet" \
  --output "${RAY_DATA_HOME}/math/math_eval_deepscalar.parquet"

echo ""
echo "=========================================="
echo "[INFO] Running Eurus Code Data Preparation..."
echo "=========================================="
python3 "${SCRIPT_DIR}/prepare_eurus_data.py" \
  --dataset "${EURUS_DATASET:-PRIME-RL/Eurus-2-RL-Data}" \
  --train-split "${EURUS_TRAIN_SPLIT:-train}" \
  --val-split "${EURUS_VAL_SPLIT:-validation}" \
  --train-output "${RAY_DATA_HOME}/eurus/eurus_code_train.parquet" \
  --val-output "${RAY_DATA_HOME}/eurus/eurus_code_val.parquet"

echo ""
echo "[SUCCESS] Data preparation complete!"
