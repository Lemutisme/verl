#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RUN_VARIANT=${RUN_VARIANT:-"pd_reward"}
export PROJECT_NAME=${PROJECT_NAME:-"deepcoder_grpo_pd_reward"}
export EXP_NAME=${EXP_NAME:-"GRPO-DeepCoder-Qwen3-4B-PDReward"}
export DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-2}

export DEEPCODER_REWARD_MODE=${DEEPCODER_REWARD_MODE:-"primal_dual"}
export DEEPCODER_USE_PRIMAL_DUAL=${DEEPCODER_USE_PRIMAL_DUAL:-true}
export DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-true}
export DEEPCODER_BETA=${DEEPCODER_BETA:-1.0}
export DEEPCODER_GAMMA=${DEEPCODER_GAMMA:-1.0}

exec "${SCRIPT_DIR}/run_grpo.sh" "$@"