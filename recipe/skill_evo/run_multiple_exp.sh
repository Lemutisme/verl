#!/usr/bin/env bash

# run_multiple_exp.sh
# Usage: bash run_multiple_exp.sh [-gpus xx]

# Default values
GPUS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -gpus)
      GPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# 1. Automatic GPU Detection
# If GPUS is not specified, find the GPU with the most free memory
if [ -z "$GPUS" ]; then
    echo "[INFO] No GPUs specified. Detecting available GPU via nvidia-smi..."
    # Query index and free memory, sort by free memory descending, take the top one
    GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -n -k 2 -r | head -n 1 | awk -F', ' '{print $1}')
    if [ -z "$GPUS" ]; then
        echo "[ERROR] Could not detect any GPUs. Please specify -gpus manually."
        exit 1
    fi
    echo "[INFO] Automatically selected GPU: $GPUS"
else
    echo "[INFO] Using user-specified GPU(s): $GPUS"
fi

# 2. Experiment Matrix
REWARDS=("ori" "new" "pd")
# Datasets for run_grpo_math.sh
MATH_DATASETS=("deepscalar" "general365")

# Paths to scripts
# run_grpo_math.sh is in the same directory
MATH_SCRIPT="./run_grpo_math.sh"
# run_grpo.sh for Code is also in the same directory
CODE_SCRIPT="./run_grpo.sh"

# 3. Main Loop
for REWARD in "${REWARDS[@]}"; do
    echo ""
    echo "################################################################"
    echo "# STARTING EXPERIMENTS FOR REWARD PRESET: ${REWARD}"
    echo "################################################################"
    echo ""

    # --- Task 1: Math (DeepScalar) ---
    echo "[RUN] Math: DeepScalar | Reward: ${REWARD}"
    bash "${MATH_SCRIPT}" -reward "${REWARD}" -dataset deepscalar -gpus "${GPUS}"

    # --- Task 2: General Reasoning (General365) ---
    echo "[RUN] General: General365 | Reward: ${REWARD}"
    bash "${MATH_SCRIPT}" -reward "${REWARD}" -dataset general365 -gpus "${GPUS}"

    # --- Task 3: Code (DeepCoder) ---
    echo "[RUN] Code: DeepCoder | Reward: ${REWARD}"
    bash "${CODE_SCRIPT}" -reward "${REWARD}" -gpus "${GPUS}"

    echo ""
    echo "[DONE] Completed experiments for reward: ${REWARD}"
    echo "################################################################"
    echo ""
done

echo "All scheduled experiments have finished."
