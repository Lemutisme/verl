# Generalized Primal-Dual Reward Shaping Framework

This repository introduces a scalable, dynamic reward shaping architecture designed to seamlessly integrate arbitrary auxiliary subrewards (e.g., reasoning elegance, computational efficiency, formatting constraints) alongside primary task accuracy in Large Language Model (LLM) Reinforcement Learning.

## 🌟 Overview

In Reinforcement Learning (RL), static reward scalarization (i.e., fixed linear combinations of subrewards) frequently suffers from scale imbalances, leading to reward collapse or stunted early-stage exploration. To mitigate this, we implement a **Generalized Primal-Dual Constrained Optimization Framework**.

This framework treats auxiliary subrewards as dual constraints, dynamically adjusting their Lagrange multipliers ($\lambda$) based on the Exponential Moving Average (EMA) of historical policy performance. 

### Key Innovations
1. **Zero-Configuration Dynamic Discovery**: The framework employs a reflection-based auto-discovery mechanism. Developers simply return arbitrary keys in a `subrewards` dictionary during evaluation. The global combiner instantaneously detects novel subrewards, initializes an isolated EMA state tracker, and subjects it to Primal-Dual regulation—all without requiring modifications to the routing logic.
2. **Adaptive Regularization**: The framework dynamically scales the impact of each subreward using adaptive step sizes ($\eta$) and targets ($\tau$) conditioned on the global primary accuracy ($S_{\text{perf}}$).
3. **Decoupled Architecture**: Evaluation environments (e.g., MBPP, DeepCoder) are strictly isolated from optimization logic. Evaluators serve purely as scoring oracles, while the `GenericRewardCombiner` acts as a centralized gradient shaper.

---

## 📐 Mathematical Formulation

### 1. Reward Composition
Given a generated trajectory, let $S_{\text{perf}} \in [0, 1]$ denote the primary accuracy (e.g., test case pass rate). Let $\mathbf{s} = \{s_1, s_2, \dots, s_k\}$ denote a set of arbitrary auxiliary subrewards. The combined scalarized reward $R$ passed to the PPO/GRPO actor is computed as:

$$
R = S_{\text{perf}} + \sum_{k} \lambda_{k} \cdot (s_k - \tau_k)
$$

Where:
- $\lambda_k \ge 0$ is the dynamically adapted Lagrange multiplier for subreward $k$.
- $\tau_k \in [\tau_{\text{min}}, \tau_{\text{max}}]$ is an adaptive target baseline, scaling linearly with the global EMA of $S_{\text{perf}}$.

*Note: The auxiliary penalty is aggressively gated. If $S_{\text{perf}}$ falls below a critical threshold (`perf_gate`), $R$ collapses to $0.0$ to ensure the policy prioritizes primary task completion over auxiliary elegance.*

### 2. Dual Update Rule (Adaptive Multipliers)
At each training step, the state tracks the exponential moving average of both the primary accuracy ($\bar{S}_{\text{perf}}$) and each subreward ($\bar{s}_k$). The multiplier $\lambda_k$ is updated via dual gradient ascent:

$$
\lambda_{k}^{(t+1)} = \text{Clip}\left( \lambda_{k}^{(t)} + \eta^{(t)} \cdot \left( \tau_k^{(t)} - \bar{s}_{k}^{(t)} \right), \ 0, \ \lambda_{\text{max}} \right)
$$

Where $\eta^{(t)}$ is an adaptive learning rate that decays over time and is governed by a sigmoidal gating function to prevent instability during early policy exploration.

---

## 🛠️ Usage & Integration

### 1. Implementing a Custom Evaluator
To introduce a new task or a new subreward constraint, simply author an evaluation function that returns a tuple of the primary accuracy and a dictionary of subrewards.

```python
# reward_score/my_custom_task.py
def compute_score(code: str, ground_truth: str, **kwargs) -> Dict[str, float]:
    acc = run_tests(code, ground_truth)
    
    if kwargs.get("return_components", False):
        return {
            "main_reward": acc,
            "subrewards": {
                "reasoning_entropy": calculate_entropy(code),
                "token_efficiency": calculate_efficiency(code)
            }
        }
    return acc
```

### 2. Executing Training
The framework is fully integrated into the training execution scripts. You can toggle between different reward shaping paradigms using the `--reward` flag.

- **Primal-Dual Optimization (Recommended)**
  Automatically activates EMA tracking and dynamically calibrates constraints.
  ```bash
  bash run_grpo.sh -reward pd -model qwen3-4b ++reward_model.reward_kwargs.tau_token_efficiency_max=0.9
  ```

- **Static Multiplier Scalarization**
  Bypasses the PD constraints and applies fixed linear weights to the subrewards ($R = S_{\text{perf}} + \sum w_k \cdot s_k$).
  ```bash
  bash run_grpo.sh -reward new -model qwen3-4b ++reward_model.reward_kwargs.weight_reasoning_entropy=1.5
  ```

- **Vanilla Baseline**
  Evaluates solely on primary accuracy ($S_{\text{perf}}$).
  ```bash
  bash run_grpo.sh -reward ori -model qwen3-4b
  ```

## 🏗️ Configuration Hyperparameters

Hyperparameters can be injected globally via Hydra overrides (`++reward_model.reward_kwargs.[param]=value`). Because of dynamic discovery, parameters for subreward `X` are automatically resolved by prefixing the key with `X`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `perf_gate` | `float` | `0.0` | Minimum global accuracy required before subrewards are integrated. |
| `weight_{X}` | `float` | `1.0` | Static coefficient applied to subreward `X` in multiplier mode. |
| `tau_{X}_min` | `float` | `0.20` | Lower bound of the adaptive baseline target for subreward `X`. |
| `tau_{X}_max` | `float` | `0.85` | Upper bound of the adaptive baseline target for subreward `X`. |
| `eta_{X}` | `float` | `0.05` | Base step size for updating the Lagrange multiplier $\lambda_X$. |
| `lambda_{X}_max` | `float` | `4.0` | Maximum permissible magnitude for $\lambda_X$. |
| `normalize_by_dual_mass` | `bool` | `False` | Whether to normalize the final reward by dividing by $(1 + \sum \lambda_k)$. |

---

## 🧭 PD-GDPO: Primal-Dual Group-Decoupled Policy Optimization

The `pd` mode above scalarizes rewards *before* GRPO normalization
($R = S_{\text{perf}} + \sum_k \lambda_k (s_k - \tau_k)$), which collapses
component-level reward geometry: once summed, distinct reward profiles can
normalize to similar advantages, and $\lambda_k$ applied before group
normalization is partially erased.

**PD-GDPO** moves primal-dual control from the *scalar reward level* to the
*component-wise advantage level*:

1. group-normalize the primary reward $r^0$ within each prompt group;
2. for each auxiliary component $k$, form correctness-gated residuals
   $c_i^k = \mathbf{1}[r_i^0 > g]\,(s_i^k - \tau_k)$ and group-normalize them
   *independently*;
3. aggregate $A_i^{\text{raw}} = \hat A_i^0 + \sum_k \rho(\lambda_k)\,\hat A_i^k$;
4. batch-whiten to get the final advantage.

Dual variables are updated once per rollout batch *after* pricing, using only
correctness-gated samples: $\lambda_k \leftarrow \Pi_{[0,\lambda_{\max}]}[\lambda_k + \eta_k(\tau_k - \widehat C_k)]$.

### Architecture

| File | Role |
|------|------|
| `pd_gdpo/controller.py` | `PrimalDualController` — centralized dual state ($\lambda_k$, $\tau_k$, EMA), updated once per batch on the driver. |
| `pd_gdpo/advantage.py` | `compute_pd_gdpo_advantage` — registered as the `pd_gdpo` advantage estimator. |
| `custom_reward.py` | In `combine_mode=pdgdpo`, emits the primary reward as `score` and each subreward as a `pdcomp__<name>` batch field (no scalarization). |

The estimator reads component scalars from `data.non_tensor_batch` keys
prefixed with `pdcomp__`. A one-line change in `verl/trainer/ppo/ray_trainer.py`
passes the full `DataProto` to any estimator that declares a `data` parameter.

### Usage

```bash
bash run_grpo.sh -reward pdgdpo -model qwen3-4b -kl loss
```

This sets `combine_mode=pdgdpo` and `algorithm.adv_estimator=pd_gdpo`. KL is
kept as an actor loss (`use_kl_loss=true`, `use_kl_in_reward=false`), so the
advantage estimator only handles reward components.

Controller hyperparameters are configured via `PDGDPO_*` environment variables
(see `run_grpo.sh -h`), e.g.:

```bash
PDGDPO_CORRECTNESS_GATE=0.0 \
PDGDPO_DEFAULT_TAU_MIN=0.2 PDGDPO_DEFAULT_TAU_MAX=0.85 \
PDGDPO_RHO_MODE=dual_mass PDGDPO_DUAL_UPDATE=additive \
bash run_grpo.sh -reward pdgdpo -model qwen3-4b
```

If no `pdcomp__*` components are present in a batch, the estimator safely falls
back to plain GRPO on the primary reward.
