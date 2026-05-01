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

# 2.1) Helper: Occupy GPU on failure
occupy_card_on_failure() {
    local exit_code=$1
    local task_name=$2
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[CRITICAL] Task '${task_name}' FAILED with exit code ${exit_code}."
        echo "[INFO] Starting GPU occupation (90% memory) to hold the card..."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
        # Only occupy the first GPU if a list is provided
        local target_gpu=$(echo $GPUS | cut -d',' -f1)
        CUDA_VISIBLE_DEVICES=${target_gpu} python3 -c "
import torch, time, sys
try:
    # Use index 0 because CUDA_VISIBLE_DEVICES is set to exactly one GPU
    device = torch.device('cuda:0')
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * 0.9)
    print(f'Successfully allocated {target_mem / 1024**3:.2f} GB on GPU ${target_gpu}')
    x = torch.empty(target_mem // 4, dtype=torch.float32, device=device)
    print('GPU is now HELD. The script will wait here for your investigation.')
    print('Kill this process (or Ctrl+C) to release the card.')
    while True: time.sleep(3600)
except Exception as e:
    print(f'Occupation failed: {e}')
    sys.exit(1)
"
        exit $exit_code
    fi
}

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
    occupy_card_on_failure $? "Math:DeepScalar:${REWARD}"

    # --- Task 2: General Reasoning (General365) ---
    echo "[RUN] General: General365 | Reward: ${REWARD}"
    bash "${MATH_SCRIPT}" -reward "${REWARD}" -dataset general365 -gpus "${GPUS}"
    occupy_card_on_failure $? "General:General365:${REWARD}"

    # --- Task 3: Code (DeepCoder) ---
    echo "[RUN] Code: DeepCoder | Reward: ${REWARD}"
    bash "${CODE_SCRIPT}" -reward "${REWARD}" -gpus "${GPUS}"
    occupy_card_on_failure $? "Code:DeepCoder:${REWARD}"

    echo ""
    echo "[DONE] Completed experiments for reward: ${REWARD}"
    echo "################################################################"
    echo ""
done

echo "All scheduled experiments have finished."
