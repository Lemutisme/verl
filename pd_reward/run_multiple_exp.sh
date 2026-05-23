#!/usr/bin/env bash

# run_multiple_exp.sh
# Usage: bash run_multiple_exp.sh [-gpus xx] [-steps N] [-reward {pdpo|gdpo|new|ori}] [-save]

# Default values
GPUS=""
STEPS="1000"
REWARD_FILTER=""
CLEANUP_RAY_VLLM=false
SAVE_MODEL=false
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

usage() {
  cat <<'EOF'
Usage:
  bash run_multiple_exp.sh [-gpus xx] [-steps N] [-reward {pdpo|gdpo|new|ori}] [-save] [--cleanup-ray-vllm]

Options:
  -gpus, --gpus             GPU ids to pass to child runs, e.g. 0 or 0,1
  -steps, --steps           Total training steps for each child run (default: 400)
  -reward, --reward         Run only one reward preset: pdpo, gdpo, new, or ori
  -save, --save             Enable model checkpoint saving for child runs
  --cleanup-ray-vllm        Stop local Ray and kill vLLM before/after tasks
  -h, --help                Show this help message

By default this script does not stop Ray or kill vLLM, so other multi-GPU jobs
on the same host are not interrupted. It also disables model checkpoint saving
unless -save is passed.
EOF
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -gpus|--gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      GPUS="$2"
      shift 2
      ;;
    -steps|--steps)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      STEPS="$2"
      shift 2
      ;;
    -reward|--reward)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 1; }
      REWARD_FILTER="$(lower "$2")"
      shift 2
      ;;
    --cleanup-ray-vllm)
      CLEANUP_RAY_VLLM=true
      shift
      ;;
    -save|--save)
      SAVE_MODEL=true
      shift
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

# 1. Experiment Matrix
REWARDS=("pdpo" "gdpo" "new" "ori")
if [ -n "$REWARD_FILTER" ]; then
    case "$REWARD_FILTER" in
        pdpo|pdpo_reward)
            REWARDS=("pdpo")
            ;;
        gdpo|gdpo_reward)
            REWARDS=("gdpo")
            ;;
        new|ori)
            REWARDS=("$REWARD_FILTER")
            ;;
        *)
            echo "[ERROR] Unsupported reward preset: ${REWARD_FILTER}" >&2
            usage
            exit 1
            ;;
    esac
fi
echo "[INFO] Reward preset(s): ${REWARDS[*]}"
echo "[INFO] Ray/vLLM cleanup enabled: ${CLEANUP_RAY_VLLM}"

MATH_SAVE_ARGS=()
if [ "$SAVE_MODEL" = true ]; then
    export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5}"
    export SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
else
    export SAVE_EVERY_STEPS="-1"
    export SAVE_BEST_CHECKPOINT="false"
    MATH_SAVE_ARGS=(--save_freq -1)
fi
echo "[INFO] Model checkpoint saving enabled: ${SAVE_MODEL}"
echo "[INFO] SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS}"
echo "[INFO] SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT}"

# 2. Automatic GPU Detection
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

# Datasets for train_math.sh
MATH_DATASETS=("deepscalar")

# Paths to scripts
MATH_SCRIPT="${DIR}/train_math.sh"
CODE_SCRIPT="${DIR}/train_code.sh"

# 2.1) Helper: Log failure and continue (replaces occupy_card_on_failure)
FAILED_TASKS=()

log_failure() {
    local exit_code=$1
    local task_name=$2
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[FAILED] Task '${task_name}' FAILED with exit code ${exit_code}."
        echo "[INFO] Continuing to next experiment..."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
        FAILED_TASKS+=("${task_name} (exit=${exit_code})")
    else
        echo "[OK] Task '${task_name}' completed successfully."
    fi
}

cleanup_ray_vllm() {
    local when=$1
    if [ "$CLEANUP_RAY_VLLM" != true ]; then
        echo "[INFO] Skipping Ray/vLLM cleanup ${when} (pass --cleanup-ray-vllm to enable)."
        return 0
    fi

    echo "[INFO] Cleaning up Ray and potential zombie vLLM processes ${when}..."
    ray stop --force >/dev/null 2>&1 || true
    pkill -f vllm >/dev/null 2>&1 || true
    sleep 3
}

# 2.2) GPU occupation helper (optional, use with -occupy flag)
occupy_gpu_until_killed() {
    echo "[INFO] Starting GPU occupation (90% memory) to hold the card..."
    local target_gpu=$(echo $GPUS | cut -d',' -f1)
    CUDA_VISIBLE_DEVICES=${target_gpu} python3 -c "
import torch, time, sys
try:
    device = torch.device('cuda:0')
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * 0.9)
    print(f'Successfully allocated {target_mem / 1024**3:.2f} GB on GPU ${target_gpu}')
    x = torch.empty(target_mem // 4, dtype=torch.float32, device=device)
    print('GPU is now HELD. Kill this process (or Ctrl+C) to release the card.')
    while True: time.sleep(3600)
except Exception as e:
    print(f'Occupation failed: {e}')
    sys.exit(1)
"
}

# 2.3) CUDA memory fragmentation prevention
# vLLM (since recent versions) explicitly checks and throws an Assertion error 
# because expandable_segments:True is incompatible with vLLM's memory pool mechanism.
# Thus we must comment it out or set it to false:
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
# Child runs inherit these defaults, and train_math.sh also passes the matching
# vLLM Hydra override. Keep them enabled for colocated actor/ref/vLLM jobs:
# without them, step0 checkpoint -> vLLM sleep-mode wake-up can fail with
# "CUDA Error: out of memory at cumem_allocator.cpp:62".
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-true}"
echo "[INFO] VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM}"
echo "[INFO] NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE}"
echo "[INFO] VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"

# 2.4) Setup Global Logging Directory
EXP_LOG_DIR="${DIR}/logs_multi_exp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${EXP_LOG_DIR}"
echo "[INFO] All stdout and stderr logs will be saved to: ${EXP_LOG_DIR}"

# 3. Infinite Loop — runs until manually stopped (Ctrl+C)
ROUND=0

while true; do
    ROUND=$((ROUND + 1))
    ROUND_FAILED_TASKS=()
    echo ""
    echo "================================================================"
    echo "  ROUND ${ROUND} — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    for DATASET in "${MATH_DATASETS[@]}"; do
        echo ""
        echo "################################################################"
        echo "# [Round ${ROUND}] BENCHMARK: ${DATASET}"
        echo "################################################################"
        echo ""

        for REWARD in "${REWARDS[@]}"; do
            echo "  --------------------------------------------------------------"
            echo "  # REWARD PRESET: ${REWARD}"
            echo "  --------------------------------------------------------------"

            cleanup_ray_vllm "before task"

            # Align with coding high-performance config
            export VLLM_GPU_UTIL=0.3
            export VLLM_MAX_NUM_SEQS=128
            export TRAIN_PROMPT_BSZ=4
            export GEN_PROMPT_BSZ=16
            export N_RESP_PER_PROMPT=4
            export TRAIN_PROMPT_MINI_BSZ=4
            export OFFLOAD=false

            TASK_OUT="${EXP_LOG_DIR}/R${ROUND}_math_${DATASET}_${REWARD}.stdout"
            TASK_ERR="${EXP_LOG_DIR}/R${ROUND}_math_${DATASET}_${REWARD}.stderr"
            echo "[RUN] Math/General: ${DATASET} | Reward: ${REWARD}"
            echo "      ➜  Stdout: ${TASK_OUT}"
            echo "      ➜  Stderr: ${TASK_ERR}"
            bash "${MATH_SCRIPT}" -reward "${REWARD}" -dataset "${DATASET}" -gpus "${GPUS}" -steps "${STEPS}" "${MATH_SAVE_ARGS[@]}" > >(tee "${TASK_OUT}") 2> >(tee "${TASK_ERR}" >&2)
            log_failure $? "Math:${DATASET}:${REWARD}"
            
            cleanup_ray_vllm "after task"
        done
    done

    # --- Task 4: Code (Eurus) ---
    echo ""
    echo "################################################################"
    echo "# [Round ${ROUND}] BENCHMARK: eurus"
    echo "################################################################"
    echo ""

    for REWARD in "${REWARDS[@]}"; do
        echo "  --------------------------------------------------------------"
        echo "  # REWARD PRESET: ${REWARD}"
        echo "  --------------------------------------------------------------"

        cleanup_ray_vllm "before task"

        # Align with coding high-performance config
        export VLLM_GPU_UTIL=0.3
        export VLLM_MAX_NUM_SEQS=128
        export TRAIN_PROMPT_BSZ=4
        export GEN_PROMPT_BSZ=16
        export N_RESP_PER_PROMPT=4
        export TRAIN_PROMPT_MINI_BSZ=4
        export OFFLOAD=false

        TASK4_OUT="${EXP_LOG_DIR}/R${ROUND}_code_eurus_${REWARD}.stdout"
        TASK4_ERR="${EXP_LOG_DIR}/R${ROUND}_code_eurus_${REWARD}.stderr"
        echo "[RUN] Code: Eurus | Reward: ${REWARD}"
        echo "      ➜  Stdout: ${TASK4_OUT}"
        echo "      ➜  Stderr: ${TASK4_ERR}"
        bash "${CODE_SCRIPT}" -reward "${REWARD}" -gpus "${GPUS}" -steps "${STEPS}" > >(tee "${TASK4_OUT}") 2> >(tee "${TASK4_ERR}" >&2)
        log_failure $? "Code:Eurus:${REWARD}"
        
        cleanup_ray_vllm "after task"
    done

    # Round Summary
    echo ""
    echo "================================================================"
    echo "  ROUND ${ROUND} SUMMARY — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
        echo "[SUCCESS] All experiments in round ${ROUND} completed successfully!"
    else
        echo "[WARNING] ${#FAILED_TASKS[@]} total experiment(s) have FAILED so far:"
        for task in "${FAILED_TASKS[@]}"; do
            echo "  - ${task}"
        done
    fi
    echo "================================================================"
    echo ""
    echo "[INFO] Starting next round in 10 seconds... (Ctrl+C to stop)"
    sleep 10
done
