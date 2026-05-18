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

Where $\eta^{(t)}$ is an adaptive learning rate that decays over time according to $\eta^{(t)} = \eta^{(0)} / (1 + 0.1 \cdot \ln(t + 1))$ to prevent updates from freezing too quickly. It is also governed by a sigmoidal gating function to prevent instability during early policy exploration.

*Note: Dual updates are deferred and aggregated to happen once per training step (instead of per-sample) to prevent catastrophic step inflation.*

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
| `lambda_{X}_max` | `float` | `0.5` | Maximum permissible magnitude for $\lambda_X$ (reduced from 4.0 to prevent sub-rewards from overwhelming the main reward). |
| `normalize_by_dual_mass` | `bool` | `True` | Whether to normalize the final reward by dividing by $(1 + \sum \lambda_k)$ (enabled by default in PD mode to prevent clipping saturation). |
