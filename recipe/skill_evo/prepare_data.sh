#!/usr/bin/env bash
# Script to prepare all required datasets for GRPO experiments
set -e

# Configuration
PYTHON="/shared/nas2/yujiz/anaconda3/envs/verl/bin/python3"
DATA_ROOT="/shared/nas2/yujiz/rl/data"
RECIPE_DIR="/shared/nas2/yujiz/rl/verl/recipe/skill_evo"

cd "${RECIPE_DIR}"

echo "=== Preparing general365 ==="
mkdir -p "${DATA_ROOT}/general365"
${PYTHON} data_preprocess/general365.py --local_dir "${DATA_ROOT}/general365"

echo "=== Preparing openr1_math ==="
mkdir -p "${DATA_ROOT}/openr1_math"
${PYTHON} data_preprocess/openr1_math_220k.py --local_dir "${DATA_ROOT}/openr1_math"

echo "=== Data preparation complete ==="
ls -R "${DATA_ROOT}/general365" "${DATA_ROOT}/openr1_math"
