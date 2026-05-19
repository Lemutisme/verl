#!/usr/bin/env bash
# Minimal launcher for PD-GDPO on a math task (GSM8K / MATH-style).
#
# Primary reward: final-answer correctness (verl default scorer).
# Auxiliary components: math_final_answer_reward,
#                      math_answer_efficiency_reward,
#                      math_consistency_reward.
#
# Override anything via env vars before invoking, e.g.:
#   MODEL_PATH=/path/to/qwen TRAIN_FILE=... TEST_FILE=... bash run_pd_gdpo_math.sh
#
# Tune dual hyper-parameters either via the Hydra config block
# (algorithm.pd_gdpo.*) or via the PDGDPO_* environment variables.

set -xeuo pipefail

project_name='PD-GDPO'
exp_name='pd-gdpo-math-qwen2.5'

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# Paths -- set these for your env.
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-1.5B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/math_dapo-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/math500.parquet"}

# Lengths / batch
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
train_prompt_bsz=128
n_resp_per_prompt=8
train_prompt_mini_bsz=32

# PD-GDPO components -- adjust to taste.  Defaults exclude
# math_final_answer_reward because the primary reward IS final-answer
# correctness (would be redundant).  Re-enable in component_keys + flip
# math_enable_final_answer_reward=true in reward_kwargs if you want it.
component_keys='[math_answer_efficiency_reward,math_consistency_reward]'

# Reward kwargs passed into compute_score; this is what flips
# collect_subrewards on for the math category.
reward_kwargs='{math_enable_sub_rewards: true}'

python3 -m recipe.pdpo.main_pdpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    algorithm.adv_estimator=pd_gdpo \
    algorithm.use_kl_in_reward=false \
    algorithm.pd_gdpo.component_keys="${component_keys}" \
    algorithm.pd_gdpo.correctness_gate=0.0 \
    algorithm.pd_gdpo.perf_lo=0.2 \
    algorithm.pd_gdpo.perf_hi=0.9 \
    algorithm.pd_gdpo.component_defaults.tau_min=0.20 \
    algorithm.pd_gdpo.component_defaults.tau_max=0.85 \
    algorithm.pd_gdpo.component_defaults.eta=0.05 \
    algorithm.pd_gdpo.component_defaults.lambda_max=2.0 \
    algorithm.pd_gdpo.rho_mode=dual_mass \
    reward.custom_reward_function.path=recipe/pdpo/custom_reward.py \
    reward.custom_reward_function.name=compute_score \
    reward.custom_reward_function.reward_kwargs="${reward_kwargs}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.logger='[console,wandb]' \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=5
