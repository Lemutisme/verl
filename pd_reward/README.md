# Multi-Reward GRPO Framework: From Reward-Level PD to Advantage-Level PDAR

This repository implements a progressive hierarchy of multi-reward integration methods for GRPO-based LLM reinforcement learning, culminating in **PDAR (Primal-Dual Advantage Regulation)** — a closed-loop advantage-level regulation mechanism.

## 🌟 Overview

Training LLMs with multiple reward signals (e.g., correctness + reasoning efficiency + format compliance) is challenging because naive reward scalarization suffers from scale imbalances and unstable advantage estimation. This framework provides four progressively more sophisticated approaches:

| Level | Method | Flag | What it does |
|-------|--------|------|-------------|
| 0 | **Vanilla** | `-reward ori` | Single reward (correctness only) |
| 1 | **Static Multiplier** | `-reward new` | Fixed-weight linear combination of subrewards |
| 2 | **Reward-Level PD** | `-reward pd` | Primal-dual constrained reward combination with adaptive $\lambda$ |
| 3 | **PDAR** | `-reward pdar` | **Advantage-level** primal-dual regulation with selective sharpness damping |

### Why PDAR? The Problem with Reward-Level PD

Reward-level PD (Level 2) combines rewards *before* GRPO normalisation:

$$R_i^{PD} = R_i^{main} + \sum_k \lambda_k R_{i,k}^{aux}$$

This is problematic because GRPO's group-relative normalisation then distorts the dual signal — changing $\lambda$ reshapes the normalisation denominator via covariance terms, creating unstable feedback loops.

PDAR (Level 3) fixes this by operating *after* independent normalisation:

$$\widetilde{A}_i = A_i^{main} + \sum_k s_k \lambda_k A_i^{aux,k}$$

Each reward channel is group-normalised independently (as in GDPO), then combined with feedback-controlled dual weights. A selective sharpness controller additionally dampens extreme within-group advantages.

---

## 📐 Mathematical Formulation

### Stage 1: Reward-Level PD (existing, `-reward pd`)

Given primary accuracy $S_{\text{perf}}$ and subrewards $\mathbf{s} = \{s_1, \dots, s_k\}$:

$$R = S_{\text{perf}} + \sum_{k} \lambda_{k} \cdot (s_k - \tau_k)$$

Dual update: $\lambda_{k}^{(t+1)} = \text{Clip}\left( \lambda_{k}^{(t)} + \eta^{(t)} \cdot ( \tau_k^{(t)} - \bar{s}_{k}^{(t)} ), \ 0, \ \lambda_{\text{max}} \right)$

### Stage 2: PDAR — Advantage-Level Regulation (new, `-reward pdar`)

#### Step 1: Decoupled Normalisation
For each reward channel $c \in \{main, aux_1, \dots, aux_K\}$, compute group-relative advantages independently:

$$A_i^c = \frac{r_i^c - \mu_G^c}{\sigma_G^c + \epsilon}$$

This is the GDPO approach — each channel's normalisation is independent of the others.

#### Step 2: Dual-Controlled Combination
Combine using feedback-controlled dual variables:

$$\widetilde{A}_i = A_i^{main} + \sum_k s_k \lambda_k A_i^{aux,k}$$

where $s_k \in \{+1, -1\}$ controls whether higher aux is better (+1) or a violation (-1).

**Constraint dual update** (based on raw auxiliary metric, not normalised advantage):

$$\lambda_c \leftarrow \text{Clip}\left(\lambda_c + \eta_c \cdot s_c \cdot (\tau_c - \bar{r}_{aux}), \ 0, \ \lambda_{c,max}\right)$$

#### Step 3: Selective Sharpness Damping
Non-linearly compress extreme within-group advantages using a bounded influence function:

$$d_i = \frac{\widetilde{A}_i - \bar{A}}{q_G + \epsilon}, \quad d_i^{stable} = \frac{d_i}{1 + \lambda_S |d_i|}$$

$$A_i^{stable} = \bar{A} + q_G \cdot d_i^{stable}$$

Properties:
- Small advantages ($|d_i| \ll 1/\lambda_S$) are nearly unchanged
- Extreme advantages ($|d_i| \gg 1/\lambda_S$) are compressed to $\sim q_G / \lambda_S$
- Sign and ranking are preserved
- When $\lambda_S = 0$, this degrades to standard GRPO

**Sharpness dual update** (slow EMA-based):

$$S_{ema} \leftarrow (1-\alpha) S_{ema} + \alpha \cdot \text{Std}(\widetilde{A})$$
$$\lambda_S \leftarrow \text{Clip}\left(\lambda_S + \eta_S \cdot (S_{ema} - \tau_S), \ 0, \ \lambda_{S,max}\right)$$

#### Key difference from Reward-Level PD

In reward-level PD, $\lambda$ interacts with group covariance inside the normalisation denominator. In PDAR, $\lambda$ is a clean linear coefficient in normalised space — each channel's normalisation is independent of $\lambda$.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  custom_reward.py                                           │
│  ┌─────────────┐   ┌──────────────────┐                     │
│  │ Main reward  │   │ Aux subrewards   │                     │
│  │ (accuracy)   │   │ (efficiency etc) │                     │
│  └──────┬───────┘   └────────┬─────────┘                     │
│         │                    │                               │
│    ─────┴────────────────────┴──────                         │
│    │  combine_mode=pdar: separate channels  │                │
│    │  combine_mode=pd: scalarize here       │                │
│    └────────────────────────────────────────┘                │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │ NaiveRewardManager      │
          │ score → token_level     │
          │ aux → reward_extra_info │
          └────────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          │ ray_trainer.py          │
          │ construct aux tensor    │
          │ call compute_advantage  │
          └────────────┬────────────┘
                       │
    ┌──────────────────┴──────────────────┐
    │ pdar_advantage.py                   │
    │ 1. Group-norm main & aux separately │
    │ 2. Combine: Ã = A_m + λ·A_a        │
    │ 3. Selective sharpness damp         │
    │ 4. Update λ_c, λ_s                 │
    └─────────────────────────────────────┘
```

---

## 🛠️ Usage & Integration

### Quick Start

```bash
# PDAR (recommended — advantage-level regulation)
bash run_grpo_math.sh -reward pdar -dataset deepscalar -model r1-1.5b

# Reward-level PD (legacy)
bash run_grpo_math.sh -reward pd -dataset deepscalar -model r1-1.5b

# Static multiplier
bash run_grpo_math.sh -reward new -model qwen3-4b

# Vanilla baseline
bash run_grpo_math.sh -reward ori -model qwen3-4b
```

### Custom PDAR Hyperparameters

Via environment variables:
```bash
PDAR_ETA_C=0.1 PDAR_TAU_S=2.0 bash run_grpo_math.sh -reward pdar -dataset gsm8k
```

Via Hydra overrides:
```bash
bash run_grpo_math.sh -reward pdar \
  ++reward_model.reward_kwargs.pdar_eta_c=0.1 \
  ++reward_model.reward_kwargs.pdar_tau_s=2.0
```

### Automated Multi-Experiment

```bash
# Runs all 4 presets (pdar, pd, new, ori) × all datasets in infinite loop
bash run_multiple_exp.sh -gpus 0 -steps 400
```

### Implementing a Custom Evaluator

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

---

## 🏗️ Configuration Hyperparameters

### Reward-Level PD (`-reward pd`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `perf_gate` | `float` | `0.0` | Minimum accuracy before subrewards are integrated |
| `weight_{X}` | `float` | `1.0` | Static weight for subreward `X` in multiplier mode |
| `tau_{X}_min` | `float` | `0.20` | Lower bound of adaptive target for subreward `X` |
| `tau_{X}_max` | `float` | `0.85` | Upper bound of adaptive target for subreward `X` |
| `eta_{X}` | `float` | `0.05` | Step size for $\lambda_X$ update |
| `lambda_{X}_max` | `float` | `0.5` | Maximum $\lambda_X$ magnitude |
| `normalize_by_dual_mass` | `bool` | `True` | Normalize by $(1 + \sum \lambda_k)$ to prevent clipping saturation |

### PDAR (`-reward pdar`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdar_eta_c` | `float` | `0.05` | Step size for constraint dual $\lambda_c$ |
| `pdar_eta_s` | `float` | `0.01` | Step size for sharpness dual $\lambda_S$ (intentionally slow) |
| `pdar_lambda_c_max` | `float` | `1.0` | Hard cap on $\lambda_c$ |
| `pdar_lambda_s_max` | `float` | `2.0` | Hard cap on $\lambda_S$ |
| `pdar_tau_c` | `float` | `0.5` | Target for mean auxiliary reward |
| `pdar_tau_s` | `float` | `1.5` | Target group advantage std (sharpness threshold) |
| `pdar_sign_c` | `float` | `1.0` | `+1` if higher aux is better, `-1` if it's a violation |
| `pdar_sharpness_ema_alpha` | `float` | `0.1` | EMA smoothing for sharpness statistic |

---

## 🗺️ Positioning vs Related Work

| Method | Normalisation | Weights | Update Geometry |
|--------|--------------|---------|-----------------|
| **GDPO** (arXiv:2601.05242) | Decoupled | Fixed (1:1) | None |
| **MO-GRPO** (arXiv:2509.22047) | Decoupled | Variance-based auto | None |
| **Constrained GRPO** (arXiv:2602.05863) | Scalarized | Lagrangian | None |
| **Reward-Level PD** (this repo, `-reward pd`) | Joint | Feedback-controlled | None |
| **PDAR** (this repo, `-reward pdar`) | Decoupled | Feedback-controlled | Selective sharpness damping |

> **Note:** PDAR does not claim formal CMDP guarantees. The dual variables serve as adaptive controllers in normalised advantage space, not raw-space Lagrange multipliers.

---

## 📁 File Structure

```
pd_reward/
├── custom_reward.py           # Unified reward entry point (dispatches by combine_mode)
├── pdar_advantage.py          # PDAR advantage estimator (registered with verl)
├── pdar_init.py               # Registration entry point
├── run_grpo_math.sh           # Math/general training script
├── run_grpo.sh                # Code training script
├── run_multiple_exp.sh        # Automated experiment runner
├── reward_score/
│   ├── primal_dual_core.py    # Reward-level PD (Stage 1)
│   ├── pdar_core.py           # PDAR core: dual state, selective damping (Stage 2)
│   ├── sub_reward/            # Subreward modules (math, coding)
│   └── ...
├── test_pdar_core.py          # Unit tests (20/20)
└── test_pdar_advantage.py     # Integration tests
```
