#!/usr/bin/env bash
# GDPO (static-weights multi-reward) baseline on GSM8K, Qwen2.5-1.5B-Instruct,
# single H200. Same primary + auxiliary components as PD-GDPO, but with
# fixed equal weights (no dual variables, no gating).
#
# 3-way comparison this script enables:
#   GRPO vs GDPO  -> does multi-reward decoupling help vs single-reward?
#   GDPO vs PD-GDPO -> does adaptive dual variables help vs fixed weights?
#
# Usage: CUDA_VISIBLE_DEVICES=7 bash recipe/pdpo/run_gdpo_gsm8k_1gpu.sh

set -xeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/PDPO}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# vLLM stability flags -- same as GRPO / PD-GDPO scripts.
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

MODEL_PATH=${MODEL_PATH:-/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
TRAIN_FILE=${TRAIN_FILE:-/root/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-/root/data/gsm8k/test.parquet}

PROJECT_NAME=${PROJECT_NAME:-pdpo-vs-grpo-gsm8k}
EXP_NAME=${EXP_NAME:-gdpo-qwen2.5-1.5b-gsm8k}
CKPT_DIR=${CKPT_DIR:-/workspace/PDPO/checkpoints/${EXP_NAME}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-2048}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}

# Reward components -- same as PD-GDPO. `score` is the primary
# correctness reward placed in non_tensor_batch by our custom_reward.py.
GDPO_KEYS='[score,math_answer_efficiency_reward,math_consistency_reward]'
GDPO_WEIGHTS='[1.0,1.0,1.0]'

# Reward kwargs forwarded into compute_score; flips math sub-reward collection on.
REWARD_KWARGS='{math_enable_sub_rewards: true}'

mkdir -p "${CKPT_DIR}"

/root/miniconda3/envs/verl/bin/python -m verl.trainer.main_ppo \
    ++ray_kwargs.ray_init.address=local \
    algorithm.adv_estimator=gdpo \
    algorithm.use_kl_in_reward=false \
    +algorithm.gdpo_reward_keys=${GDPO_KEYS} \
    +algorithm.gdpo_reward_weights=${GDPO_WEIGHTS} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_remove_padding=false \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    reward.custom_reward_function.path=recipe/pdpo/custom_reward.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs="${REWARD_KWARGS}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.logger='[console,tensorboard]' \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
