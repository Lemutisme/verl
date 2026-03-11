#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RUN_VARIANT=${RUN_VARIANT:-"ori"}
export PROJECT_NAME=${PROJECT_NAME:-"deepcoder_grpo_ori"}
export EXP_NAME=${EXP_NAME:-"GRPO-DeepCoder-Qwen3-4B-Ori"}
export DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-0}

# DeepCoder does not currently have a dedicated upstream "original" reward module.
# Use pure correctness as the baseline by disabling extra shaping terms.
export DEEPCODER_REWARD_MODE=${DEEPCODER_REWARD_MODE:-"action_thought"}
export DEEPCODER_USE_PRIMAL_DUAL=${DEEPCODER_USE_PRIMAL_DUAL:-false}
export DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-false}
export DEEPCODER_BETA=${DEEPCODER_BETA:-0.0}
export DEEPCODER_GAMMA=${DEEPCODER_GAMMA:-0.0}

exec "${SCRIPT_DIR}/run_grpo.sh" "$@"