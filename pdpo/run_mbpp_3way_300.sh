#!/usr/bin/env bash
# Sequential 3-way MBPP comparison, 300 steps each.
# Runs GRPO -> GDPO-gated -> PD-GDPO on the same GPU one after another.
# Total ~16h. Logs to logs/{name}_mbpp_300.log per run; checkpoints
# under checkpoints/{name}-qwen3-4b-mbpp-300/.

set -euo pipefail

cd /workspace/PDPO

# All three runs share these env-var overrides.
export TOTAL_TRAINING_STEPS=300

run_one() {
    local label="$1"
    local script="$2"
    local exp_name="$3"
    local logf="/workspace/PDPO/logs/${label}_mbpp_300.log"

    echo "=== launching ${label} (${exp_name}) at $(date) ==="
    EXP_NAME="${exp_name}" CKPT_DIR="/workspace/PDPO/checkpoints/${exp_name}" \
        CUDA_VISIBLE_DEVICES=7 \
        bash "${script}" > "${logf}" 2>&1
    local rc=$?
    echo "=== ${label} exited rc=${rc} at $(date) ==="
    return $rc
}

run_one "grpo"        recipe/pdpo/run_grpo_mbpp_1gpu.sh        grpo-qwen3-4b-mbpp-300        || true
run_one "gdpo-gated"  recipe/pdpo/run_gdpo_gated_mbpp_1gpu.sh  gdpo-gated-qwen3-4b-mbpp-300  || true
run_one "pdpo"        recipe/pdpo/run_pdpo_mbpp_1gpu.sh        pdpo-qwen3-4b-mbpp-300        || true

echo "=== all 3 runs complete at $(date) ==="
