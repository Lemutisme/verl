# PDPO Reward and Advantage Experiments

This directory contains the reward functions, advantage estimators, launch scripts, and tests for the PDPO experiments on math and coding tasks.

The current main method is **PDPO (Process-Distance Policy Optimization)**:

> Keep the original task reward as the optimization target, and use process-distance subrewards only to estimate group-relative advantages when the original reward is too sparse or tied.

This is intentionally different from simply adding more reward terms. PDPO changes the **advantage estimation geometry**, not the semantic target of the task.

## Method Summary

| Method | Flag | Reward Used For Score | Advantage Estimator | Main Role |
|---|---|---|---|---|
| Vanilla GRPO | `-reward ori` | original reward only | `grpo` | baseline |
| Static subreward mix | `-reward new` | scalarized main + aux | `grpo` | fixed reward shaping |
| GDPO baseline | `-reward gdpo` | main and aux channels exported separately | `gdpo` | fixed-weight per-channel normalization baseline |
| **PDPO** | `-reward pdpo` | original/main reward as anchor, aux exported separately | `pdpo` | correctness-safe, reliability-aware process advantage estimation |

`run_multiple_exp.sh` now defaults to the active matrix:

```bash
REWARDS=("pdpo" "gdpo" "new" "ori")
```

Run PDPO explicitly with:

```bash
bash run_multiple_exp.sh -gpus 5 -reward pdpo
```

## Is PDPO Fundamental?

Short answer: **yes, relative to GRPO/GDPO/reward mixing, PDPO is the more fundamental formulation for our setting**. The reason is that the real bottleneck is not only reward design; it is sparse-outcome **advantage identifiability**.

### What GRPO Fails To See

GRPO computes group-relative advantages from the final scalar reward:

$$
A_i = \frac{R_i - \mu_G(R)}{\sigma_G(R) + \epsilon}
$$

For sparse correctness rewards, many groups are flat:

- all responses wrong: `R = [0, 0, 0, 0]`
- all responses partially similar
- all responses receive the same clipped/normalized score

Then GRPO has no within-group learning signal, even when one wrong answer has much better process quality than another.

### What Reward Mixing Gets Wrong

Reward-level methods solve the flat group problem by changing the reward:

$$
R_i^{mix} = R_i^{main} + \sum_k w_k r_{i,k}^{aux}
$$

This creates signal, but it also changes the target. A model can learn to optimize process-looking behavior even when that behavior is not reliably improving final correctness. In addition, scalarizing before GRPO normalization couples reward weights with the group mean/std, making the effect of each subreward unstable across groups.

### What GDPO Fixes, And What It Leaves Open

GDPO's key insight is correct: normalize each reward channel independently before combining:

$$
A_i^{GDPO} = \sum_k w_k \frac{r_{i,k} - \mu_G(r_k)}{\sigma_G(r_k) + \epsilon}
$$

This avoids scale domination between channels. But fixed weights still do not know whether a group already has a reliable correctness ordering. If an auxiliary channel is anti-correlated in a group, GDPO can fight the final reward.

### PDPO's Core Move

PDPO uses the original reward as the anchor and uses auxiliary process signals only in advantage space:

$$
A_i^{main} = \text{GroupNorm}(R_i^{main})
$$

For each process channel:

$$
A_{i,k}^{aux} = \text{GroupNorm}(r_{i,k}^{aux})
$$

Then:

$$
A_i^{PDPO}
= A_i^{main}
+ \lambda_{aux} \cdot \beta_G \sum_k w_k \rho_k m_{G,k} A_{i,k}^{aux}
$$

where:

- `m_{G,k}=1` only when auxiliary channel `k` has non-trivial group variance.
- `rho_k` is a per-channel reliability scale from recent mixed-outcome groups.
- `beta_G = beta_same` when the main reward is flat inside the group.
- `beta_G = beta_tie` when the main reward already distinguishes samples.
- Math PDPO gates non-answer auxiliary channels with `math_answer_extractability_reward` by default, so traces without an
  extractable answer do not get full process-advantage credit.
- Mixed-outcome groups use strict correctness safety: aux can rank samples within the same main-reward bucket, but cannot move a lower-main-reward sample above a higher-main-reward sample.

Default behavior:

```bash
PDPO_BETA_SAME=0.70   # aux can guide all-wrong / tied groups, but less aggressively
PDPO_BETA_TIE=0.20    # aux is only a weak tie-breaker when correctness varies
PDPO_LAMBDA_AUX=0.70
PDPO_MIN_AUX_STD=1e-6
PDPO_MIN_MAIN_STD=1e-6
PDPO_CORRECTNESS_SAFE=true
PDPO_RELIABILITY_ENABLED=true
```

This makes PDPO more fundamental than plain reward shaping because it addresses the actual failure mode:

> outcome reward defines what we want; process reward estimates which samples should get gradient when outcome reward cannot rank them.

### Limits

PDPO is not a formal convergence guarantee. It now protects the main correctness ordering in mixed-outcome groups and downweights anti-aligned process channels, but it still depends on having at least some directionally useful process signals in flat all-wrong/all-correct groups.

## Implementation

### Reward Path

`custom_reward.py` returns:

- `score`: the scalar reward used as the main token-level reward.
- `main_reward`: original/main task reward.
- flattened subreward keys such as `math_step_arithmetic_validity_reward` or `coding_compiler_runtime_feedback`.
- `aux_reward_combined`: retained for legacy compatibility.

For PDPO, the trainer ignores `aux_reward_combined` and reads the flattened per-channel subrewards.

### Trainer Path

[ray_trainer.py](/shared/nas2/yujiz/rl/verl/verl/trainer/ppo/ray_trainer.py) registers the local `pdpo` estimator. New runs should use only `pdpo`, `new`, or `ori`.

For `pdpo`, it extracts numeric aux channels from `data.non_tensor_batch`:

- included: `math_*`, `coding_*`
- excluded: `score`, `main_reward`, `original_reward`, `acc`, `partial_pass_rate`, `aux_reward_combined`, metadata fields

### Advantage Path

[pdpo_advantage.py](/shared/nas2/yujiz/rl/verl/pd_reward/pdpo_advantage.py) implements:

1. group-normalize main reward.
2. group-normalize each auxiliary channel independently.
3. skip auxiliary channels with no group variance.
4. estimate per-channel reliability from mixed-outcome groups.
5. update per-channel safety duals when aux channels are high on wrong samples or fail the correct-minus-wrong margin.
6. preserve main-reward ordering in mixed-outcome groups with strict no-crossing.
7. use auxiliary guidance freely only when main reward is flat, and only within same-main buckets otherwise.
8. apply the existing selective sharpness damping controller.

Metrics emitted include:

- `pdpo/main_reward_mean`
- `pdpo/aux_mean`
- `pdpo/active_channels`
- `pdpo/active_group_count`
- `pdpo/group_adv_std_before`
- `pdpo/group_adv_std_after`
- `pdpo/beta_tie`
- `pdpo/beta_same`
- `pdpo/lambda_aux`
- `pdpo/lambda_aux_effective`
- `pdpo/correctness_safe_clamp_count`
- `pdpo/correctness_margin_min`
- per-channel `pdpo/channel/<name>/mean`
- per-channel `pdpo/channel/<name>/weight`
- per-channel `pdpo/channel/<name>/effective_weight`
- per-channel `pdpo/channel/<name>/preference_weight`
- per-channel `pdpo/channel/<name>/reliability`
- per-channel `pdpo/channel/<name>/safety_dual_mu`
- per-channel `pdpo/channel/<name>/safety_dual_scale`
- per-channel `pdpo/channel/<name>/safety_dual_violation`
- per-channel `pdpo/channel/<name>/safety_dual_pressure`
- per-channel `pdpo/channel/<name>/safety_dual_updated`
- per-channel `pdpo/channel/<name>/wrong_high_rate`

## Usage

### Math / General

```bash
bash train_math.sh -reward pdpo -dataset deepscalar -gpus 5
```

DeepScaleR is the default math training dataset for current sweeps. Its eval file is:

```text
/shared/nas2/yujiz/rl/data/math/math_eval_deepscalar.parquet
```

That eval suite combines the existing math master eval with General365 test and OlympiadBench. Override with `DEEPSCALAR_VAL_FILE` when needed.

### Coding

```bash
bash train_code.sh -reward pdpo -gpus 5
```

The coding launcher defaults to Eurus-2-RL prepared files:

```text
/shared/nas2/yujiz/rl/data/eurus/eurus_code_train.parquet
/shared/nas2/yujiz/rl/data/eurus/eurus_code_val.parquet
```

Override with `EURUS_TRAIN_FILE` and `EURUS_VAL_FILE`. Legacy DeepCoder files remain a fallback when Eurus files are absent.

Prepare the default data with:

```bash
bash data_preprocess/prepare_data.sh
```

### Multi-Experiment Runner

```bash
bash run_multiple_exp.sh -gpus 5 -reward pdpo
```

The multi-experiment runner currently sweeps math on DeepScaleR and code on Eurus.

## Recommended Defaults

### Math Subrewards

The executable preset currently disables the older saturated rewards by default and enables the more local process signals:

```bash
MATH_ENABLE_FINAL_ANSWER_REWARD=false
MATH_ENABLE_ANSWER_EFFICIENCY_REWARD=false
MATH_ENABLE_CONSISTENCY_REWARD=false
MATH_ENABLE_EXECUTABLE_UNIT_PASS_RATE_REWARD=false

MATH_ENABLE_STEP_ARITHMETIC_VALIDITY_REWARD=true
MATH_WEIGHT_STEP_ARITHMETIC_VALIDITY_REWARD=0.35
MATH_ENABLE_PREFIX_CONSISTENCY_REWARD=true
MATH_WEIGHT_PREFIX_CONSISTENCY_REWARD=0.15
MATH_ENABLE_TRACE_EFFICIENCY_REWARD=true
MATH_WEIGHT_TRACE_EFFICIENCY_REWARD=0.35
MATH_ENABLE_ANSWER_EXTRACTABILITY_REWARD=true
MATH_WEIGHT_ANSWER_EXTRACTABILITY_REWARD=0.15
```

### Coding Subrewards

The coding path uses one general executable reward implementation for MBPP-style assert tests and Eurus/DeepCoder-style
stdin/stdout tests. Dataset-specific thought/action rewards are not part of the active aux channels.

```bash
CODING_ENABLE_CODE_EXTRACTABILITY_REWARD=true
CODING_WEIGHT_CODE_EXTRACTABILITY_REWARD=0.15
CODING_ENABLE_SYNTAX_VALIDITY_REWARD=true
CODING_WEIGHT_SYNTAX_VALIDITY_REWARD=0.25
CODING_ENABLE_COMPILER_RUNTIME_FEEDBACK=true
CODING_WEIGHT_COMPILER_RUNTIME_FEEDBACK=0.30
CODING_ENABLE_EXECUTED_TOKEN_CREDIT=false
CODING_WEIGHT_EXECUTED_TOKEN_CREDIT=0.0
CODING_ENABLE_STATIC_ANALYSIS_REWARD=false
CODING_WEIGHT_STATIC_ANALYSIS_REWARD=0.0
CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD=false
CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD=0.0
```

## Hyperparameters

### PDPO

| Env var | Hydra key | Default | Meaning |
|---|---|---:|---|
| `PDPO_BETA_SAME` | `reward_model.reward_kwargs.pdpo_beta_same` | `0.70` | Aux strength when main reward is flat in group |
| `PDPO_BETA_TIE` | `reward_model.reward_kwargs.pdpo_beta_tie` | `0.20` | Aux strength when main reward already varies |
| `PDPO_LAMBDA_AUX` | `reward_model.reward_kwargs.pdpo_lambda_aux` | `0.70` | Global multiplier for aux advantages |
| `PDPO_LAMBDA_AUX_START` | `reward_model.reward_kwargs.pdpo_lambda_aux_start` | `0.30` | Initial aux multiplier during warmup |
| `PDPO_LAMBDA_AUX_WARMUP_STEPS` | `reward_model.reward_kwargs.pdpo_lambda_aux_warmup_steps` | `100` | Internal PDPO steps to ramp aux multiplier to `PDPO_LAMBDA_AUX` |
| `PDPO_MIN_AUX_STD` | `reward_model.reward_kwargs.pdpo_min_aux_std` | `1e-6` | Minimum group std for an aux channel to be active |
| `PDPO_MIN_MAIN_STD` | `reward_model.reward_kwargs.pdpo_min_main_std` | `1e-6` | Minimum main-reward group std to treat main as informative |
| `PDPO_ANSWER_GATE_CHANNEL` | `reward_model.reward_kwargs.pdpo_answer_gate_channel` | math: `math_answer_extractability_reward`, code: `coding_code_extractability_reward` | Channel used as the answer/code extractability gate |
| `PDPO_ANSWER_GATE_MIN` | `reward_model.reward_kwargs.pdpo_answer_gate_min` | `0.5` | Minimum answer-extractability score needed for full non-answer aux credit |
| `PDPO_ANSWER_GATE_CLOSED_SCALE` | `reward_model.reward_kwargs.pdpo_answer_gate_closed_scale` | `0.0` | Multiplier for non-answer aux channels when the answer gate is closed |
| `PDPO_ANSWER_GATE_AS_CONSTRAINT` | `reward_model.reward_kwargs.pdpo_answer_gate_as_constraint` | `true` | Use answer extractability as a gate/constraint instead of a direct preference reward |
| `PDPO_ANSWER_GATE_PREFERENCE_SCALE` | `reward_model.reward_kwargs.pdpo_answer_gate_preference_scale` | `0.0` | Residual preference weight for the answer-gate channel when used as a constraint |
| `PDPO_CORRECTNESS_SAFE` | `reward_model.reward_kwargs.pdpo_correctness_safe` | `true` | Preserve main-reward ordering in mixed-outcome groups |
| `PDPO_CORRECTNESS_MARGIN` | `reward_model.reward_kwargs.pdpo_correctness_margin` | `1e-3` | Minimum gap between adjacent main-reward buckets after aux shaping |
| `PDPO_RELIABILITY_ENABLED` | `reward_model.reward_kwargs.pdpo_reliability_enabled` | `true` | Enable per-channel reliability scaling |
| `PDPO_RELIABILITY_EMA_ALPHA` | `reward_model.reward_kwargs.pdpo_reliability_ema_alpha` | `0.05` | EMA update rate for reliability |
| `PDPO_RELIABILITY_MIN_SCALE` | `reward_model.reward_kwargs.pdpo_reliability_min_scale` | `0.0` | Lower bound for reliability scale |
| `PDPO_RELIABILITY_MAX_SCALE` | `reward_model.reward_kwargs.pdpo_reliability_max_scale` | `1.0` | Upper bound for reliability scale |
| `PDPO_RELIABILITY_TARGET_MARGIN` | `reward_model.reward_kwargs.pdpo_reliability_target_margin` | `0.02` | Correct-minus-wrong aux gap that reaches full reliability |
| `PDPO_RELIABILITY_NEGATIVE_TOLERANCE` | `reward_model.reward_kwargs.pdpo_reliability_negative_tolerance` | `0.02` | Anti-correlation tolerance before strong downweighting |
| `PDPO_RELIABILITY_WRONG_HIGH_THRESHOLD` | `reward_model.reward_kwargs.pdpo_reliability_wrong_high_threshold` | `0.30` | Aux score treated as high on wrong samples |
| `PDPO_RELIABILITY_WRONG_HIGH_TARGET` | `reward_model.reward_kwargs.pdpo_reliability_wrong_high_target` | `0.20` | Wrong high-rate tolerated before downweighting |
| `PDPO_RELIABILITY_MIN_COMPARABLE_GROUPS` | `reward_model.reward_kwargs.pdpo_reliability_min_comparable_groups` | `4` | Minimum comparable prompt groups before updating reliability EMA |
| `PDPO_RELIABILITY_WRONG_HIGH_SMOOTHING` | `reward_model.reward_kwargs.pdpo_reliability_wrong_high_smoothing` | `1.0` | Beta-style smoothing mass for wrong-high-rate estimates |
| `PDPO_SAFETY_DUAL_ENABLED` | `reward_model.reward_kwargs.pdpo_safety_dual_enabled` | `true` | Enable PDPO-internal per-channel safety dual scaling |
| `PDPO_SAFETY_DUAL_ETA` | `reward_model.reward_kwargs.pdpo_safety_dual_eta` | `0.05` | Safety dual update rate |
| `PDPO_SAFETY_DUAL_MU_MAX` | `reward_model.reward_kwargs.pdpo_safety_dual_mu_max` | `6.0` | Max per-channel safety dual value |
| `PDPO_SAFETY_DUAL_DECAY` | `reward_model.reward_kwargs.pdpo_safety_dual_decay` | `0.0` | Optional recovery decay for safety dual values |
| `PDPO_SAFETY_DUAL_TARGET_MARGIN` | `reward_model.reward_kwargs.pdpo_safety_dual_target_margin` | `0.02` | Required correct-minus-wrong aux margin before no dual penalty |
| `PDPO_SAFETY_DUAL_WRONG_HIGH_TARGET` | `reward_model.reward_kwargs.pdpo_safety_dual_wrong_high_target` | `0.20` | Wrong high-rate tolerated before safety dual penalty |
| `PDPO_SAFETY_DUAL_MIN_COMPARABLE_GROUPS` | `reward_model.reward_kwargs.pdpo_safety_dual_min_comparable_groups` | `4` | Minimum comparable prompt groups before primal-dual update |
| `PDPO_SAFETY_DUAL_EMA_ALPHA` | `reward_model.reward_kwargs.pdpo_safety_dual_ema_alpha` | `0.10` | EMA rate for signed constraint pressure |
| `PDPO_SAFETY_DUAL_RECOVERY_SCALE` | `reward_model.reward_kwargs.pdpo_safety_dual_recovery_scale` | `0.25` | Multiplier for negative pressure that recovers dual values |
| `PDPO_ETA_S` | `reward_model.reward_kwargs.pdpo_eta_s` | `0.01` | Sharpness dual step size |
| `PDPO_LAMBDA_S_MAX` | `reward_model.reward_kwargs.pdpo_lambda_s_max` | `2.0` | Max damping strength |
| `PDPO_TAU_S` | `reward_model.reward_kwargs.pdpo_tau_s` | `1.5` | Target group advantage std |
| `PDPO_SHARPNESS_EMA_ALPHA` | `reward_model.reward_kwargs.pdpo_sharpness_ema_alpha` | `0.1` | EMA smoothing |

Example:

```bash
PDPO_BETA_TIE=0.10 PDPO_BETA_SAME=0.70 \
  bash train_math.sh -reward pdpo -dataset general365 -gpus 5
```

## Positioning

| Method | Main Problem Solved | Remaining Problem |
|---|---|---|
| GRPO | Simple outcome-relative policy optimization | no signal in flat sparse-reward groups |
| Static reward mixing | fixed process reward shaping | changes objective; scalarization interacts with GRPO normalization |
| GDPO | decoupled per-channel normalization | fixed weights can fight correctness |
| **PDPO** | correctness-safe per-channel process advantage estimation | still needs useful process signals in flat groups |

## Files

```text
pd_reward/
├── custom_reward.py              # Reward entry point and flattened reward extras
├── pdpo_advantage.py             # PDPO advantage estimator
├── pdpo_init.py                  # Registers local PDPO estimator
├── train_math.sh                 # Math/general training launcher
├── train_code.sh                 # Coding training launcher
├── run_multiple_exp.sh           # Multi-experiment launcher
├── data_preprocess/
│   └── prepare_eurus_data.py      # Eurus-2-RL coding train/eval preparation
├── reward_score/
│   ├── coding_executable_reward.py # Shared coding executable reward path
│   └── sub_reward/               # Math/coding subreward modules
└── test/
    ├── test_pdpo_advantage.py
    └── test_reward_revisions.py
```

## Verification

Run the local test suite from the repo root:

```bash
cd /shared/nas2/yujiz/rl/verl
source /shared/nas2/yujiz/anaconda3/etc/profile.d/conda.sh
conda activate verl
python -m pytest -q pd_reward
bash -n pd_reward/train_code.sh pd_reward/train_math.sh pd_reward/run_multiple_exp.sh
```
