# PDPO / PDAR Reward and Advantage Experiments

This directory contains the reward functions, advantage estimators, launch scripts, and tests for the PD/PDAR/PDPO experiments on math and coding tasks.

The current main method is **PDPO (Process-Distance Policy Optimization)**:

> Keep the original task reward as the optimization target, and use process-distance subrewards only to estimate group-relative advantages when the original reward is too sparse or tied.

This is intentionally different from simply adding more reward terms. PDPO changes the **advantage estimation geometry**, not the semantic target of the task.

## Method Summary

| Method | Flag | Reward Used For Score | Advantage Estimator | Main Role |
|---|---|---|---|---|
| Vanilla GRPO | `-reward ori` | original reward only | `grpo` | baseline |
| Static subreward mix | `-reward new` | scalarized main + aux | `grpo` | fixed reward shaping |
| Reward-level PD | `-reward pd` | adaptive scalarized main + aux | `grpo` | primal-dual reward shaping |
| PDAR | `-reward pdar` | main + aux exported separately | `pdar` | advantage-level aux regulation |
| PDAR-ORI | `-reward pdar-ori` | original reward only | `pdar` | original reward with PDAR geometry |
| **PDPO** | `-reward pdpo` | original/main reward as anchor, aux exported separately | `pdpo` | correctness-anchored process advantage estimation |

`run_multiple_exp.sh` still defaults to the older matrix:

```bash
REWARDS=("pdar" "pd" "new" "ori")
```

Run PDPO explicitly with:

```bash
bash run_multiple_exp.sh -gpus 5 -reward pdpo
```

## Is PDPO Fundamental?

Short answer: **yes, relative to GRPO/GDPO/PDAR, PDPO is the more fundamental formulation for our setting**. The reason is that the real bottleneck is not only reward design; it is sparse-outcome **advantage identifiability**.

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
+ \lambda_{aux} \cdot \beta_G \sum_k w_k m_{G,k} A_{i,k}^{aux}
$$

where:

- `m_{G,k}=1` only when auxiliary channel `k` has non-trivial group variance.
- `beta_G = beta_same` when the main reward is flat inside the group.
- `beta_G = beta_tie` when the main reward already distinguishes samples.

Default behavior:

```bash
PDPO_BETA_SAME=1.00   # aux can guide all-wrong / tied groups
PDPO_BETA_TIE=0.20    # aux is only a weak tie-breaker when correctness varies
PDPO_LAMBDA_AUX=1.00
PDPO_MIN_AUX_STD=1e-6
PDPO_MIN_MAIN_STD=1e-6
```

This makes PDPO more fundamental than plain reward shaping because it addresses the actual failure mode:

> outcome reward defines what we want; process reward estimates which samples should get gradient when outcome reward cannot rank them.

### Limits

PDPO is not a formal guarantee. It relies on process subrewards being directionally useful. If a subreward is noisy, saturated, or anti-correlated with final correctness, PDPO can still inject bad gradient. The current implementation uses group variance masks and fixed channel weights; future improvements can add adaptive reliability/correlation gates.

## Implementation

### Reward Path

`custom_reward.py` returns:

- `score`: the scalar reward used as the main token-level reward.
- `main_reward`: original/main task reward.
- flattened subreward keys such as `math_step_arithmetic_validity_reward` or `coding_compiler_runtime_feedback`.
- `aux_reward_combined`: retained for PDAR compatibility.

For PDPO, the trainer ignores `aux_reward_combined` and reads the flattened per-channel subrewards.

### Trainer Path

[ray_trainer.py](/shared/nas2/yujiz/rl/verl/verl/trainer/ppo/ray_trainer.py) registers local estimators for:

- `pdar`
- `pdpo`

For `pdpo`, it extracts numeric aux channels from `data.non_tensor_batch`:

- included: `math_*`, `coding_*`, `thought`, `action`
- excluded: `score`, `main_reward`, `original_reward`, `acc`, `partial_pass_rate`, `aux_reward_combined`, metadata fields

### Advantage Path

[pdpo_advantage.py](/shared/nas2/yujiz/rl/verl/pd_reward/pdpo_advantage.py) implements:

1. group-normalize main reward.
2. group-normalize each auxiliary channel independently.
3. skip auxiliary channels with no group variance.
4. use strong auxiliary guidance only when main reward is flat.
5. use weak auxiliary guidance when main reward already has signal.
6. apply the existing selective sharpness damping controller.

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
- per-channel `pdpo/channel/<name>/mean`
- per-channel `pdpo/channel/<name>/weight`

## Usage

### Math / General

```bash
bash run_grpo_math.sh -reward pdpo -dataset gsm8k -gpus 5
bash run_grpo_math.sh -reward pdpo -dataset deepscalar -gpus 5
bash run_grpo_math.sh -reward pdpo -dataset general365 -gpus 5
```

### Coding

```bash
bash run_grpo.sh -reward pdpo -gpus 5
```

### Multi-Experiment Runner

```bash
bash run_multiple_exp.sh -gpus 5 -reward pdpo
```

`pdpo` is intentionally not added to the default matrix to avoid silently expanding long-running experiment sweeps.

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
MATH_WEIGHT_PREFIX_CONSISTENCY_REWARD=0.25
MATH_ENABLE_TRACE_EFFICIENCY_REWARD=true
MATH_WEIGHT_TRACE_EFFICIENCY_REWARD=0.25
MATH_ENABLE_ANSWER_EXTRACTABILITY_REWARD=true
MATH_WEIGHT_ANSWER_EXTRACTABILITY_REWARD=0.15
```

### Coding Subrewards

The current coding defaults avoid the saturated/thought-action rewards and keep the more direct execution signals:

```bash
DEEPCODER_ENABLE_THOUGHT=false
DEEPCODER_BETA=0.0
DEEPCODER_GAMMA=0.0

CODING_ENABLE_COMPILER_RUNTIME_FEEDBACK=true
CODING_WEIGHT_COMPILER_RUNTIME_FEEDBACK=0.30
CODING_ENABLE_EXECUTED_TOKEN_CREDIT=true
CODING_WEIGHT_EXECUTED_TOKEN_CREDIT=0.20
CODING_ENABLE_STATIC_ANALYSIS_REWARD=false
CODING_WEIGHT_STATIC_ANALYSIS_REWARD=0.0
CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD=false
CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD=0.0
```

## Hyperparameters

### PDPO

| Env var | Hydra key | Default | Meaning |
|---|---|---:|---|
| `PDPO_BETA_SAME` | `reward_model.reward_kwargs.pdpo_beta_same` | `1.00` | Aux strength when main reward is flat in group |
| `PDPO_BETA_TIE` | `reward_model.reward_kwargs.pdpo_beta_tie` | `0.20` | Aux strength when main reward already varies |
| `PDPO_LAMBDA_AUX` | `reward_model.reward_kwargs.pdpo_lambda_aux` | `1.00` | Global multiplier for aux advantages |
| `PDPO_MIN_AUX_STD` | `reward_model.reward_kwargs.pdpo_min_aux_std` | `1e-6` | Minimum group std for an aux channel to be active |
| `PDPO_MIN_MAIN_STD` | `reward_model.reward_kwargs.pdpo_min_main_std` | `1e-6` | Minimum main-reward group std to treat main as informative |

PDPO also reuses sharpness damping knobs:

| Env var | Hydra key | Default | Meaning |
|---|---|---:|---|
| `PDAR_ETA_S` | `reward_model.reward_kwargs.pdar_eta_s` | `0.01` | Sharpness dual step size |
| `PDAR_LAMBDA_S_MAX` | `reward_model.reward_kwargs.pdar_lambda_s_max` | `2.0` | Max damping strength |
| `PDAR_TAU_S` | `reward_model.reward_kwargs.pdar_tau_s` | `1.5` | Target group advantage std |
| `PDAR_SHARPNESS_EMA_ALPHA` | `reward_model.reward_kwargs.pdar_sharpness_ema_alpha` | `0.1` | EMA smoothing |

Example:

```bash
PDPO_BETA_TIE=0.10 PDPO_BETA_SAME=0.80 \
  bash run_grpo_math.sh -reward pdpo -dataset general365 -gpus 5
```

### PDAR

| Env var | Hydra key | Default | Meaning |
|---|---|---:|---|
| `PDAR_ETA_C` | `reward_model.reward_kwargs.pdar_eta_c` | `0.05` | Constraint dual step size |
| `PDAR_LAMBDA_C_MAX` | `reward_model.reward_kwargs.pdar_lambda_c_max` | `1.0` | Max aux dual |
| `PDAR_TAU_C` | `reward_model.reward_kwargs.pdar_tau_c` | `0.5` | Target mean aux reward |
| `PDAR_SIGN_C` | `reward_model.reward_kwargs.pdar_sign_c` | `1.0` | Direction of aux constraint |

## Positioning

| Method | Main Problem Solved | Remaining Problem |
|---|---|---|
| GRPO | Simple outcome-relative policy optimization | no signal in flat sparse-reward groups |
| Reward-level PD | adaptive reward shaping | changes objective; scalarization interacts with GRPO normalization |
| GDPO | decoupled per-channel normalization | fixed weights can fight correctness |
| PDAR | advantage-level aux regulation with damping | current aux path is mostly one combined aux signal |
| **PDPO** | correctness-anchored process advantage estimation | still needs reliable process signals |

## Files

```text
pd_reward/
├── custom_reward.py              # Reward entry point and flattened reward extras
├── pdar_advantage.py             # PDAR advantage estimator
├── pdpo_advantage.py             # PDPO advantage estimator
├── pdar_init.py                  # Registers pdar and pdpo
├── run_grpo_math.sh              # Math/general training launcher
├── run_grpo.sh                   # Coding training launcher
├── run_multiple_exp.sh           # Multi-experiment launcher
├── reward_score/
│   ├── primal_dual_core.py       # Reward-level PD logic
│   ├── pdar_core.py              # Shared group norm and sharpness damping helpers
│   └── sub_reward/               # Math/coding subreward modules
└── test/
    ├── test_pdar_advantage.py
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
bash -n pd_reward/run_grpo.sh pd_reward/run_grpo_math.sh pd_reward/run_multiple_exp.sh
```
