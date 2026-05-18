#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONDA_SH=${CONDA_SH:-"/shared/nas2/yujiz/anaconda3/etc/profile.d/conda.sh"}

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
echo "[INFO] Running Code Data Preparation..."
echo "=========================================="
python3 "${SCRIPT_DIR}/prepare_code_eval.py"

echo ""
echo "[SUCCESS] Data preparation complete!"
