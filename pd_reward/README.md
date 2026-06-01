# PDPO Reward and Advantage Experiments

This directory contains the reward functions, advantage estimators, launch scripts, and tests for the PDPO experiments on math and coding tasks.

The current main method is **PDPO-Core (Process-Distance Policy Optimization)**:

> Keep the original task reward as the optimization target. Use process-distance subrewards only when the main reward cannot rank samples, or optionally inside equal-main-reward buckets.

This is intentionally different from simply adding more reward terms. PDPO changes the **advantage estimation geometry**, not the semantic target of the task. The current implementation is main-first and lexicographic: auxiliary rewards are surrogate signals, not objectives that trade off against correctness.

## Method Summary

| Method | Flag | Reward Used For Score | Advantage Estimator | Main Role |
|---|---|---|---|---|
| Vanilla GRPO | `-reward ori` | original reward only | `grpo` | baseline |
| Static subreward mix | `-reward new` | scalarized main + aux | `grpo` | fixed reward shaping |
| GDPO baseline | `-reward gdpo` | main and aux channels exported separately | `gdpo` | fixed-weight per-channel normalization baseline |
| **PDPO-Core** | `-reward pdpo` | original/main reward as anchor, aux exported separately | `pdpo` | main-first, reliability-aware process advantage estimation |

`run_multiple_exp.sh` now defaults to the active matrix:

```bash
REWARDS=("pdpo" "gdpo" "new" "ori")
```

Run PDPO explicitly with:

```bash
bash run_multiple_exp.sh -gpus 5 -reward pdpo
```

## Is PDPO Fundamental?

Short answer: **yes, relative to GRPO/GDPO/reward mixing, PDPO is the more fundamental formulation for our setting**. The reason is that the real bottleneck is not only reward design; it is sparse-outcome **advantage identifiability under a fixed correctness objective**.

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

### PDPO-Core's Core Move

PDPO-Core uses the original reward as the anchor and uses auxiliary process signals only in advantage space:

$$
A_i^{main} = \text{GroupNorm}(R_i^{main})
$$

For each process channel:

$$
A_{i,k}^{aux} = \text{GroupNorm}(r_{i,k}^{aux})
$$

Each auxiliary channel receives a reliability and dual-controlled scale:

$$
\tilde w_k = w_k \cdot \rho_k \cdot \exp(-\mu_k)
$$

where:

- auxiliary channel `k` contributes only when it has non-trivial group variance.
- `rho_k` is a per-channel reliability scale estimated from pairwise correct-vs-wrong alignment in mixed-outcome groups.
- `mu_k` is a per-channel safety dual variable. It rises when a channel inverts correct-vs-wrong pairs or is high on wrong samples, and can recover when the channel is safe again.
- Math PDPO gates non-answer auxiliary channels with `math_answer_extractability_reward` by default. That channel is treated as a gate/constraint, not a direct preference reward.

The auxiliary component is lexicographic:

$$
C_i =
\begin{cases}
\beta_{same}\sum_k \tilde w_k A_{i,k}^{aux}, & \text{if the group has flat main reward} \\
\beta_{tie}\sum_k \tilde w_k (A_{i,k}^{aux} - \bar A_{B,k}^{aux}), & \text{inside equal-main bucket } B \\
0, & \text{across different main-reward buckets}
\end{cases}
$$

Then:

$$
A_i^{PDPO} = A_i^{main} + \lambda_{aux}^{eff} C_i
$$

With the default `PDPO_BETA_TIE=0.0`, mixed-outcome groups use main reward only. Auxiliary channels are still used there to estimate reliability and dual pressure, but they do not change the advantage unless the user explicitly enables same-main-bucket tie-breaking.

Default behavior:

```bash
PDPO_BETA_SAME=0.70   # aux can guide all-wrong / tied groups, but less aggressively
PDPO_BETA_TIE=0.0     # mixed groups are main-only by default
PDPO_LAMBDA_AUX=0.70
PDPO_LAMBDA_AUX_START=0.30
PDPO_LAMBDA_AUX_WARMUP_STEPS=100
PDPO_MIN_AUX_STD=1e-6
PDPO_MIN_MAIN_STD=1e-6
PDPO_CORRECTNESS_SAFE=true
PDPO_RELIABILITY_ENABLED=true
PDPO_SAFETY_DUAL_ENABLED=true
```

This makes PDPO-Core more fundamental than plain reward shaping because it addresses the actual failure mode:

> outcome reward defines what we want; process reward estimates which samples should get gradient when outcome reward cannot rank them.

### What Primal-Dual Means Here

PDPO currently has real internal dual variables, but it is not a full constrained RL saddle-point solver.

Implemented dual controllers:

- Per-channel safety dual `mu_k`: updated from signed constraint pressure based on correct-vs-wrong margin, pairwise inversion rate, and wrong-high rate. It scales channels by `exp(-mu_k)`.
- Sharpness dual `lambda_s`: updated from group advantage std and used by the selective damping controller.

Not implemented yet:

- A global correctness dual that lowers `lambda_aux_eff` when rolling train/eval correctness drops below a target.
- A formal convergence guarantee for the full policy optimization problem.

So the current method is best described as **main-first PDPO with per-channel primal-dual safety control**, not full primal-dual RL.

### Literature-Aware Claim Boundary

The broad idea of primal-dual advantage optimization is not new. Standard constrained policy optimization uses a Lagrangian form such as:

$$
A_L = A_{reward} - \sum_i \lambda_i A_{cost_i}
$$

with projected dual updates:

$$
\lambda_i \leftarrow [\lambda_i + \eta \cdot \text{constraint_violation}_i]_+
$$

GAE supplies the usual low-variance advantage estimator; CPO, RCPO, NPG-PD, PPO-Lagrangian, Safe RLHF, and constrained GRPO already cover reward/cost advantage scalarization and dual updates. Therefore, PDPO should not be positioned as "the first primal-dual advantage estimator."

The sharper research claim is:

> Typed primal-dual process control for critic-free reasoning RL: separate format constraints, correctness-aligned process rewards, and efficiency costs; route them through different dual controllers; and compare strict Lagrangian scalarization with decoupled process shaping under GRPO-style grouped advantages.

Relevant prior directions:

- GAE: advantage estimation substrate, not a constraint method. <https://arxiv.org/abs/1506.02438>
- CPO / RCPO / NPG-PD: classical CMDP reward-cost decomposition and dual policy optimization. <https://arxiv.org/abs/1705.10528>, <https://arxiv.org/abs/1805.11074>, <https://proceedings.neurips.cc/paper_files/paper/2020/hash/5f7695debd8cde8db5abcb9f161b49ea-Abstract.html>
- PID Lagrangian: dual updates can overshoot or oscillate, motivating EMA, caps, and budget normalization. <https://arxiv.org/abs/2007.03964>
- Safe RLHF: LLM reward/cost decoupling with Lagrangian balancing. <https://arxiv.org/abs/2310.12773>
- Constrained GRPO: warns that component-wise normalization can distort Lagrangian tradeoffs, motivating scalarize-before-normalize comparisons. <https://arxiv.org/abs/2602.05863>
- GDPO / PAPO / PRPO: process and multi-reward normalization for reasoning RL, motivating decoupled process advantages and outcome anchoring. <https://arxiv.org/abs/2601.05242>, <https://arxiv.org/abs/2603.26535>, <https://arxiv.org/abs/2601.07182>

## Research Questions

### RQ1: Scalarize Before Normalize Or Normalize Per Channel?

Constrained GRPO suggests strict Lagrangian semantics should scalarize reward and costs before the grouped normalization step:

$$
A = \text{GroupNorm}(R_{main} + \sum_k \alpha_k P_k - \sum_j \lambda_j C_j)
$$

Current PDPO does the opposite:

$$
A = A_{main} + \sum_k w_k \text{GroupNorm}(P_k)
$$

This preserves each process signal's resolution but weakens strict Lagrangian interpretation because each channel's standard deviation is normalized away. The experimental question is whether strict scalarization improves constraint satisfaction and final accuracy, or whether decoupled normalization is better for sparse reasoning rewards.

### RQ2: Do Reasoning Subrewards Need Different Dual Semantics?

Process rewards are not interchangeable:

- `math_answer_extractability_reward` is a format lower-bound constraint. Wrong answers should still be parseable, so correctness-aligned wrong-high penalties can incorrectly suppress it.
- `math_step_arithmetic_validity_reward` and `math_prefix_consistency_reward` are correctness-aligned process rewards. They should be suppressed when they are high on wrong samples or invert correct-vs-wrong pairs.
- `math_trace_efficiency_reward` is closer to an efficiency cost or cap. Strong positive reward can favor short shallow traces, so it should be bounded or converted to an overlong cost.
- response length and truncation are explicit costs, not positive process rewards.

The experimental question is whether typed routing beats one shared reliability/safety controller for all channels.

### RQ3: Can Critic-Free Dual Control Work With Small GRPO Groups?

We only have grouped rollouts and no learned cost critic. Dual updates must estimate constraint violation from the current rollout batch. This creates noisy, low-sample dual signals, especially when the number of comparable correct-vs-wrong groups is small.

The experimental question is whether EMA, minimum comparable-group thresholds, caps, and budget normalization are enough to make dual tracking useful on one 80GB GPU.

### RQ4: How Do We Prevent Auxiliary Signal Starvation?

Current PDPO is stable but conservative: low `pg_clipfrac`, low KL, and small effective answer-gate preference weight indicate that auxiliary signals may be too weak to move the policy. Need-duals can amplify unmet constraints:

$$
\lambda^{need}_k \leftarrow \text{clip}(\lambda^{need}_k + \eta(t_k - \text{EMA}[m_k]), 0, \lambda^{max}_k)
$$

where `m_k` is a channel-specific satisfaction metric. The experimental question is whether need-duals improve format/process metrics and final accuracy without destabilizing KL, clip fraction, length, or correctness.

### RQ5: When Do We Need PID-Style Dual Stabilization?

Plain integral dual updates may oscillate. The first implementation should use EMA, caps, recovery decay, and total auxiliary budget normalization. If need-duals oscillate or overcorrect, add proportional or PID-style terms and compare against the simpler controller.

## Observed Root Causes And Current Fix Status

The 2026-05 PDPO run showed late-stage validation degradation despite stable-looking training metrics. The current typed-controller implementation is intended to address the following root causes:

| Root cause | Current mechanism | Expected coverage |
|---|---|---|
| Answer gate was suppressed by PDPO itself | `math_answer_extractability_reward` and `coding_code_extractability_reward` are routed as `format_constraint` channels. By default, format constraints bypass correctness-derived reliability and safety-dual suppression. | High. This directly prevents parseability from being treated as dangerous just because wrong answers are also parseable. |
| Safety dual became a stale suppressor | `PDPO_SAFETY_DUAL_DECAY` now decays stale `mu` even when comparable groups are below the update threshold. Format constraints also get a positive `need_dual` amplifier. | Medium-high. It fixes stale penalties and adds positive pressure for unmet format constraints, but process-channel safety duals are still suppressive rather than full CMDP Lagrangians. |
| Process reward can point the wrong way in flat all-wrong groups | Aux weights are budget-normalized with `PDPO_AUX_BUDGET`; `math_trace_efficiency_reward` is capped by `PDPO_EFFICIENCY_COST_WEIGHT_CAP`; format constraints are separated from correctness-aligned process rewards. | Medium. This reduces harmful process bias, but `PDPO_BETA_SAME` still controls how much process reward can shape flat groups. |
| Decoupled normalization conflicts with strict Lagrangian semantics | The current branch keeps decoupled per-channel normalization, but logs role-level weights, pre-budget weights, budget scale, need dual, and safety dual metrics. | Low. This is not solved until a scalarize-before-normalize branch is added and compared. |
| Policy movement was too weak while biased aux accumulated | Answer-gate preference is no longer near-zero, trace efficiency is capped, and total aux mass is budgeted. | Medium. Directional bias should improve, but KL, `pg_clipfrac`, and grad norm still need to be monitored in the next run. |

Promotion criteria for the typed-controller defaults:

- answer extractability `budgeted_preference_weight` should be meaningfully above the previous near-zero value (`~0.004`).
- `need_dual_lambda` should rise when extractability is below target and stop rising when the target is met.
- `pdpo/budget_scale` should stay interpretable and avoid sustained aux-weight explosions.
- hard-source validation accuracy should not show the previous late-stage AIME/general365 slide.
- `pg_clipfrac`, KL, grad norm, response length, and truncation metrics should not spike.

Remaining research gap: strict constrained optimization still requires a `PDPO_SCALARIZATION_MODE=lagrangian` branch that scalarizes reward/process/cost terms before group normalization.

### Limits

PDPO is not a formal convergence guarantee. It structurally prevents aux from crossing main correctness buckets and downweights anti-aligned process channels, but it still depends on having at least some directionally useful process signals in flat all-wrong/all-correct groups. If flat-group aux signals are not predictive of eventual correctness, PDPO-Core will correctly avoid corrupting mixed groups but may still fail to improve over `ori`.

## Implementation

### Reward Path

`custom_reward.py` returns:

- `score`: the scalar reward used as the main token-level reward.
- `main_reward`: original/main task reward.
- flattened subreward keys such as `math_step_arithmetic_validity_reward` or `coding_compiler_runtime_feedback`.
- `aux_reward_combined`: retained for legacy compatibility.

For PDPO, the trainer ignores `aux_reward_combined` and reads the flattened per-channel subrewards.

### Trainer Path

[ray_trainer.py](/shared/nas2/yujiz/rl/verl/verl/trainer/ppo/ray_trainer.py) registers the local `pdpo` estimator. New reward sweeps should use the active matrix: `pdpo`, `gdpo`, `new`, and `ori`.

For `pdpo`, it extracts numeric aux channels from `data.non_tensor_batch`:

- included: `math_*`, `coding_*`
- excluded: `score`, `main_reward`, `original_reward`, `acc`, `partial_pass_rate`, `aux_reward_combined`, metadata fields

### Advantage Path

[pdpo_advantage.py](/shared/nas2/yujiz/rl/verl/pd_reward/pdpo_advantage.py) implements:

1. group-normalize main reward.
2. group-normalize each auxiliary channel independently.
3. skip auxiliary channels with no group variance.
4. estimate per-channel reliability from pairwise correct-vs-wrong alignment in mixed-outcome groups.
5. update per-channel safety duals when aux channels invert correct-vs-wrong pairs or fail the correct-minus-wrong margin.
6. build a lexicographic aux component: flat-main groups may use aux freely; mixed groups may only use aux inside equal-main buckets.
7. preserve main-reward ordering in mixed-outcome groups with strict no-crossing.
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
- per-channel `pdpo/channel/<name>/pairwise_alignment_rate`
- per-channel `pdpo/channel/<name>/pairwise_inversion_rate`

## Code Roadmap For The Research Questions

### Stage 1: Add Typed Channel Routing

Extend `PDPOConfig` and `PDPOState` with explicit channel roles:

```text
format_constraint: answer/code extractability, no-final, valid final marker
process_reward: arithmetic validity, prefix consistency, compiler/runtime feedback
efficiency_cost: overlong response, post-answer verbosity, trace efficiency cap
```

Implementation sketch:

- Add env/Hydra controls such as `PDPO_FORMAT_CHANNELS`, `PDPO_PROCESS_CHANNELS`, and `PDPO_EFFICIENCY_COST_CHANNELS`.
- Keep current channel behavior as the default for backward compatibility.
- Exempt `format_constraint` channels from correctness-aligned wrong-high safety suppression unless explicitly enabled.
- Emit `pdpo/channel/<name>/role` and role-level aggregate metrics.

### Stage 2: Add Need-Duals For Unmet Positive Constraints

Add a second per-channel dual state:

```text
channel_need_dual[name]
channel_need_metric_ema[name]
```

For positive lower-bound constraints:

```text
violation = target - EMA(metric)
lambda_need = clip(lambda_need + eta * violation, 0, lambda_need_max)
```

Use it as an adaptive amplifier:

```text
preference_weight = base_weight * reliability * safety_scale * (1 + lambda_need)
```

Then apply budget normalization:

```text
if sum(preference_weight) > aux_budget:
    preference_weight *= aux_budget / sum(preference_weight)
```

Initial target candidates:

- answer extractability mean: `>= 0.75`
- answer gate open ratio: `>= 0.70`
- no-final ratio: `<= 0.02`
- train response clip ratio: `<= 0.10`

### Stage 3: Convert Efficiency To Cost

Do not treat trace efficiency as an unrestricted positive reward. Add a cost-style path:

```text
cost_overlong = max(0, response_tokens - target_tokens) / target_tokens
A_total = A_main + A_process - lambda_len * A_cost_overlong
```

For the decoupled branch, this can be implemented as `-lambda_len * GroupNorm(cost_overlong)`. For the strict branch, it should be scalarized before group normalization.

### Stage 4: Add Strict Lagrangian Scalarization Branch

Introduce an experiment switch:

```text
PDPO_SCALARIZATION_MODE=decoupled   # current behavior
PDPO_SCALARIZATION_MODE=lagrangian  # scalarize before group normalize
```

The `lagrangian` branch should construct one scalar per response before grouped normalization:

```text
S_i = R_i^{main}
    + sum_k alpha_k P_{i,k}
    + sum_f lambda_need_f F_{i,f}
    - sum_c lambda_cost_c C_{i,c}

A_i = GroupNorm(S_i)
```

This branch is the strict CMDP-style ablation. The current branch remains the decoupled process-shaping ablation.

### Stage 5: Add Controller Stabilization Ablations

Keep the first controller simple:

- EMA-smoothed metrics
- projected non-negative duals
- per-channel caps
- total auxiliary budget normalization
- minimum comparable-group thresholds

Only add PID-style terms if metrics show oscillation:

- repeated overshoot/undershoot in `lambda_need`
- alternating high/low no-final ratio
- KL or `pg_clipfrac` spikes after constraint recovery

## Validation Plan For The Research Questions

### Offline Checks Before Training

Run deterministic unit and synthetic tests before launching GPU jobs:

- `test_pdpo_advantage.py`: current behavior remains unchanged when new flags are disabled.
- Synthetic flat-main groups: need-dual increases unmet format channels and creates non-zero aux advantage.
- Synthetic mixed-outcome groups: correctness-safe still prevents aux from ranking wrong samples above correct samples.
- Safety channel tests: anti-aligned process rewards increase `safety_dual_mu` and reduce effective weight.
- Format channel tests: high wrong-sample extractability does not trigger safety suppression when routed as `format_constraint`.
- Budget tests: total preference mass never exceeds `PDPO_AUX_BUDGET`.
- Scalarization tests: `lagrangian` mode normalizes only after scalarizing reward/process/cost terms.

### Short GPU Probes

Use 20-50 step probes before long runs. Compare against the current defaults:

```bash
bash run_multiple_exp.sh -gpus 7 -reward pdpo
```

Minimum probe matrix:

| Run | Purpose |
|---|---|
| current PDPO | baseline |
| answer need-dual only | test RQ4 with lowest risk |
| typed routing + need-dual | test RQ2/RQ4 |
| typed routing + need-dual + budget | test stability |
| lagrangian scalarization | test RQ1 strict constrained branch |
| decoupled typed PDPO | test RQ1 decoupled branch |

### Metrics That Decide Each RQ

RQ1 needs:

- final val macro/all acc
- source-level AIME24/AIME25/general365 acc
- constraint satisfaction by source
- KL, `pg_clipfrac`, grad norm, response clip ratio
- comparison of `decoupled` versus `lagrangian` at matched token budget and eval length

RQ2 needs:

- per-role effective/preference weight
- `format_constraint` no-final and extractability
- `process_reward` pairwise alignment, inversion, wrong-high rate
- `efficiency_cost` length and truncation metrics

RQ3 needs:

- comparable group count
- dual update rate
- dual variance across steps
- robustness across seeds or at least adjacent repeated probes

RQ4 needs:

- answer-gate open ratio
- no-final ratio
- `math_answer_extractability_reward/mean`
- train response clip ratio
- policy movement metrics: KL, `pg_clipfrac`, grad norm
- final correctness not decreasing while format metrics improve

RQ5 needs:

- oscillation in dual values
- oscillation in constraint metrics
- KL or length spikes after dual changes
- comparison of plain EMA dual against PID-style controller only if oscillation appears

### Promotion Criteria

Do not promote a new controller to default unless it passes all of:

- no decrease larger than 1-2 pp in recent-window val macro accuracy versus current PDPO.
- lower no-final or truncation-sensitive failure rate on hard sources.
- no persistent KL, clipfrac, or grad spikes.
- no sustained increase in response clip ratio.
- interpretable dual traces: unmet constraints increase lambda; satisfied constraints stop increasing or recover.

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
MATH_WEIGHT_TRACE_EFFICIENCY_REWARD=0.10
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
| `PDPO_BETA_TIE` | `reward_model.reward_kwargs.pdpo_beta_tie` | `0.0` | Optional aux tie-break strength inside equal-main buckets of mixed groups |
| `PDPO_LAMBDA_AUX` | `reward_model.reward_kwargs.pdpo_lambda_aux` | `0.70` | Global multiplier for aux advantages |
| `PDPO_LAMBDA_AUX_START` | `reward_model.reward_kwargs.pdpo_lambda_aux_start` | `0.30` | Initial aux multiplier during warmup |
| `PDPO_LAMBDA_AUX_WARMUP_STEPS` | `reward_model.reward_kwargs.pdpo_lambda_aux_warmup_steps` | `100` | Internal PDPO steps to ramp aux multiplier to `PDPO_LAMBDA_AUX` |
| `PDPO_MIN_AUX_STD` | `reward_model.reward_kwargs.pdpo_min_aux_std` | `1e-6` | Minimum group std for an aux channel to be active |
| `PDPO_MIN_MAIN_STD` | `reward_model.reward_kwargs.pdpo_min_main_std` | `1e-6` | Minimum main-reward group std to treat main as informative |
| `PDPO_ANSWER_GATE_CHANNEL` | `reward_model.reward_kwargs.pdpo_answer_gate_channel` | math: `math_answer_extractability_reward`, code: `coding_code_extractability_reward` | Channel used as the answer/code extractability gate |
| `PDPO_ANSWER_GATE_MIN` | `reward_model.reward_kwargs.pdpo_answer_gate_min` | `0.5` | Minimum answer-extractability score needed for full non-answer aux credit |
| `PDPO_ANSWER_GATE_CLOSED_SCALE` | `reward_model.reward_kwargs.pdpo_answer_gate_closed_scale` | `0.0` | Multiplier for non-answer aux channels when the answer gate is closed |
| `PDPO_ANSWER_GATE_AS_CONSTRAINT` | `reward_model.reward_kwargs.pdpo_answer_gate_as_constraint` | `true` | Use answer extractability as a gate/constraint instead of a direct preference reward |
| `PDPO_ANSWER_GATE_PREFERENCE_SCALE` | `reward_model.reward_kwargs.pdpo_answer_gate_preference_scale` | `0.3` | Residual preference weight for the answer-gate channel when used as a constraint |
| `PDPO_FORMAT_CONSTRAINT_CHANNELS` | `reward_model.reward_kwargs.pdpo_format_constraint_channels` | `math_answer_extractability_reward,coding_code_extractability_reward` | Channels treated as format lower-bound constraints rather than correctness-aligned process rewards |
| `PDPO_EFFICIENCY_COST_CHANNELS` | `reward_model.reward_kwargs.pdpo_efficiency_cost_channels` | `math_trace_efficiency_reward` | Channels tagged as efficiency/cost-like diagnostics |
| `PDPO_EFFICIENCY_COST_WEIGHT_CAP` | `reward_model.reward_kwargs.pdpo_efficiency_cost_weight_cap` | `0.10` | Per-channel preference-weight cap for efficiency/cost-like channels |
| `PDPO_FORMAT_CONSTRAINT_RELIABILITY_ENABLED` | `reward_model.reward_kwargs.pdpo_format_constraint_reliability_enabled` | `false` | Allow correctness-derived reliability scaling on format constraints |
| `PDPO_FORMAT_CONSTRAINT_SAFETY_ENABLED` | `reward_model.reward_kwargs.pdpo_format_constraint_safety_enabled` | `false` | Allow correctness-derived safety-dual suppression on format constraints |
| `PDPO_CORRECTNESS_SAFE` | `reward_model.reward_kwargs.pdpo_correctness_safe` | `true` | Preserve main-reward ordering in mixed-outcome groups |
| `PDPO_CORRECTNESS_MARGIN` | `reward_model.reward_kwargs.pdpo_correctness_margin` | `1e-3` | Minimum gap between adjacent main-reward buckets after aux shaping |
| `PDPO_AUX_BUDGET` | `reward_model.reward_kwargs.pdpo_aux_budget` | `1.0` | Total preference-weight budget after reliability, safety, and need-dual scaling |
| `PDPO_AUX_BUDGET_NORMALIZE` | `reward_model.reward_kwargs.pdpo_aux_budget_normalize` | `true` | Normalize active aux preference weights to stay within `PDPO_AUX_BUDGET` |
| `PDPO_RELIABILITY_ENABLED` | `reward_model.reward_kwargs.pdpo_reliability_enabled` | `true` | Enable per-channel reliability scaling |
| `PDPO_RELIABILITY_EMA_ALPHA` | `reward_model.reward_kwargs.pdpo_reliability_ema_alpha` | `0.05` | EMA update rate for reliability |
| `PDPO_RELIABILITY_MIN_SCALE` | `reward_model.reward_kwargs.pdpo_reliability_min_scale` | `0.0` | Lower bound for reliability scale |
| `PDPO_RELIABILITY_MAX_SCALE` | `reward_model.reward_kwargs.pdpo_reliability_max_scale` | `1.0` | Upper bound for reliability scale |
| `PDPO_RELIABILITY_TARGET_MARGIN` | `reward_model.reward_kwargs.pdpo_reliability_target_margin` | `0.02` | Correct-minus-wrong aux gap that reaches full reliability |
| `PDPO_RELIABILITY_NEGATIVE_TOLERANCE` | `reward_model.reward_kwargs.pdpo_reliability_negative_tolerance` | `0.02` | Anti-correlation tolerance before strong downweighting |
| `PDPO_RELIABILITY_WRONG_HIGH_THRESHOLD` | `reward_model.reward_kwargs.pdpo_reliability_wrong_high_threshold` | `0.30` | Aux score treated as high on wrong samples |
| `PDPO_RELIABILITY_WRONG_HIGH_TARGET` | `reward_model.reward_kwargs.pdpo_reliability_wrong_high_target` | `0.20` | Wrong high-rate tolerated before downweighting |
| `PDPO_RELIABILITY_PAIRWISE_TARGET` | `reward_model.reward_kwargs.pdpo_reliability_pairwise_target` | `0.55` | Pairwise alignment rate needed for full channel reliability |
| `PDPO_RELIABILITY_INVERSION_TARGET` | `reward_model.reward_kwargs.pdpo_reliability_inversion_target` | `0.20` | Pairwise inversion rate tolerated before reliability penalty |
| `PDPO_RELIABILITY_MIN_COMPARABLE_GROUPS` | `reward_model.reward_kwargs.pdpo_reliability_min_comparable_groups` | `4` | Minimum comparable prompt groups before updating reliability EMA |
| `PDPO_RELIABILITY_WRONG_HIGH_SMOOTHING` | `reward_model.reward_kwargs.pdpo_reliability_wrong_high_smoothing` | `1.0` | Beta-style smoothing mass for wrong-high-rate estimates |
| `PDPO_SAFETY_DUAL_ENABLED` | `reward_model.reward_kwargs.pdpo_safety_dual_enabled` | `true` | Enable PDPO-internal per-channel safety dual scaling |
| `PDPO_SAFETY_DUAL_ETA` | `reward_model.reward_kwargs.pdpo_safety_dual_eta` | `0.05` | Safety dual update rate |
| `PDPO_SAFETY_DUAL_MU_MAX` | `reward_model.reward_kwargs.pdpo_safety_dual_mu_max` | `6.0` | Max per-channel safety dual value |
| `PDPO_SAFETY_DUAL_DECAY` | `reward_model.reward_kwargs.pdpo_safety_dual_decay` | `0.02` | Optional recovery decay for safety dual values |
| `PDPO_SAFETY_DUAL_TARGET_MARGIN` | `reward_model.reward_kwargs.pdpo_safety_dual_target_margin` | `0.02` | Required correct-minus-wrong aux margin before no dual penalty |
| `PDPO_SAFETY_DUAL_WRONG_HIGH_TARGET` | `reward_model.reward_kwargs.pdpo_safety_dual_wrong_high_target` | `0.20` | Wrong high-rate tolerated before safety dual penalty |
| `PDPO_SAFETY_DUAL_INVERSION_TARGET` | `reward_model.reward_kwargs.pdpo_safety_dual_inversion_target` | `0.20` | Pairwise inversion rate tolerated before safety dual pressure |
| `PDPO_SAFETY_DUAL_MIN_COMPARABLE_GROUPS` | `reward_model.reward_kwargs.pdpo_safety_dual_min_comparable_groups` | `2` | Minimum comparable prompt groups before primal-dual update |
| `PDPO_SAFETY_DUAL_EMA_ALPHA` | `reward_model.reward_kwargs.pdpo_safety_dual_ema_alpha` | `0.10` | EMA rate for signed constraint pressure |
| `PDPO_SAFETY_DUAL_RECOVERY_SCALE` | `reward_model.reward_kwargs.pdpo_safety_dual_recovery_scale` | `0.25` | Multiplier for negative pressure that recovers dual values |
| `PDPO_NEED_DUAL_ENABLED` | `reward_model.reward_kwargs.pdpo_need_dual_enabled` | `true` | Enable adaptive amplification for unmet format constraints |
| `PDPO_NEED_DUAL_ETA` | `reward_model.reward_kwargs.pdpo_need_dual_eta` | `0.05` | Need-dual update rate |
| `PDPO_NEED_DUAL_TARGET` | `reward_model.reward_kwargs.pdpo_need_dual_target` | `0.75` | Target mean satisfaction for format constraints |
| `PDPO_NEED_DUAL_MAX` | `reward_model.reward_kwargs.pdpo_need_dual_max` | `2.0` | Maximum need-dual multiplier, used as `1 + lambda_need` |
| `PDPO_NEED_DUAL_EMA_ALPHA` | `reward_model.reward_kwargs.pdpo_need_dual_ema_alpha` | `0.10` | EMA rate for format-constraint satisfaction metrics |
| `PDPO_ETA_S` | `reward_model.reward_kwargs.pdpo_eta_s` | `0.01` | Sharpness dual step size |
| `PDPO_LAMBDA_S_MAX` | `reward_model.reward_kwargs.pdpo_lambda_s_max` | `2.0` | Max damping strength |
| `PDPO_TAU_S` | `reward_model.reward_kwargs.pdpo_tau_s` | `1.5` | Target group advantage std |
| `PDPO_SHARPNESS_EMA_ALPHA` | `reward_model.reward_kwargs.pdpo_sharpness_ema_alpha` | `0.1` | EMA smoothing |

Example:

```bash
PDPO_BETA_TIE=0.0 PDPO_BETA_SAME=0.70 \
  bash train_math.sh -reward pdpo -dataset general365 -gpus 5
```

### Long-Context PDPO Defaults

The math launcher defaults to 6k training responses and 12k validation
responses:

```bash
MAX_RESPONSE_LENGTH=6144
EVAL_MAX_RESPONSE_LENGTH=12288
EVAL_EVERY_STEPS=5
VLLM_MAX_NUM_SEQS=128
TRAIN_PROMPT_BSZ=4
TRAIN_PROMPT_MINI_BSZ=4
GEN_PROMPT_BSZ=16
PPO_CLIP_RATIO=0.2
PPO_CLIP_RATIO_LOW=0.2
PPO_CLIP_RATIO_HIGH=0.3
PDPO_ANSWER_GATE_PREFERENCE_SCALE=0.3
PDPO_FORMAT_CONSTRAINT_CHANNELS=math_answer_extractability_reward,coding_code_extractability_reward
PDPO_EFFICIENCY_COST_CHANNELS=math_trace_efficiency_reward
PDPO_EFFICIENCY_COST_WEIGHT_CAP=0.10
PDPO_FORMAT_CONSTRAINT_RELIABILITY_ENABLED=false
PDPO_FORMAT_CONSTRAINT_SAFETY_ENABLED=false
PDPO_AUX_BUDGET=1.0
PDPO_AUX_BUDGET_NORMALIZE=true
PDPO_SAFETY_DUAL_MIN_COMPARABLE_GROUPS=2
PDPO_SAFETY_DUAL_DECAY=0.02
PDPO_NEED_DUAL_ENABLED=true
PDPO_NEED_DUAL_TARGET=0.75
PDPO_NEED_DUAL_MAX=2.0
MATH_WEIGHT_TRACE_EFFICIENCY_REWARD=0.10
```

Hydra note: comma-containing channel lists must be quoted in launcher overrides. `train_math.sh` passes these as `key='value1,value2'`; otherwise Hydra treats `value1,value2` as a sweep and raises `ConfigCompositionException: Ambiguous value for argument`.

Default PDPO launch:

```bash
cd /shared/nas2/yujiz/rl/verl/pd_reward
bash run_multiple_exp.sh -gpus 6 -reward pdpo
```

Fallback command if the card is fragmented or another process leaves too little
memory. This keeps training at 4k and only lengthens validation to 10k:

```bash
cd /shared/nas2/yujiz/rl/verl/pd_reward && \
MAX_RESPONSE_LENGTH=4096 \
EVAL_MAX_RESPONSE_LENGTH=10240 \
EVAL_EVERY_STEPS=5 \
VLLM_MAX_NUM_SEQS=32 \
TRAIN_PROMPT_BSZ=4 \
TRAIN_PROMPT_MINI_BSZ=4 \
GEN_PROMPT_BSZ=16 \
bash run_multiple_exp.sh -gpus 6 -reward pdpo
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
