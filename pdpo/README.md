# PD-GDPO: Primal-Dual Group-Decoupled Policy Optimization

Recipe for multi-reward RL fine-tuning where auxiliary process rewards
are treated as **constraints with adaptive Lagrange multipliers**, and
dual control is applied at the **advantage** level rather than the
scalar-reward level.

## Why

Standard GRPO / scalar reward shaping multiplies auxiliary rewards by
fixed weights and sums them before group normalization:

    R_i = r_i^0 + Σ_k w_k · s_i^k          # then GRPO

That has two issues:

1. **Scale erasure inside a group.** GRPO standardizes by per-group std.
   Because λ_k (or w_k) is constant across the G samples in a prompt
   group, the multiplication is *fully* absorbed by the normalization
   inside that group. Whatever weight you put on the auxiliary signal
   has no effect on the *relative* advantage between samples of the
   same prompt.
2. **Component collapse.** Two components with opposite per-group
   signals can cancel in the scalar reward, but should produce additive
   *directional* advantages once each is z-scored separately.

GDPO ([NVlabs/GDPO](https://github.com/NVlabs/GDPO)) fixes (2) by
group-normalizing each reward component independently and then
aggregating with fixed weights. PD-GDPO extends this by replacing those
fixed weights with **dual variables** that move in response to
constraint violations:

    A_i^0          = GRPO-normalize(r_i^0)
    c_i^k          = 1[r_i^0 > g] · (s_i^k - τ_k)              # gated residual
    A_i^k          = GRPO-normalize(c_i^k)                     # per-component
    A_i^raw        = A_i^0 + Σ_k ρ(λ_k) · A_i^k                # dual-weighted
    A_i            = BatchWhiten(A_i^raw)                      # final adv
    λ_k ← Π_[0,λ_max] [ λ_k + η_k · (τ_k - Ĉ_k) ]              # post-pricing
    τ_k(t)         = τ_min + (τ_max - τ_min) · ratio(ema_primary)

The primal objective is the primary reward. Auxiliary process rewards
are constraints `C_k ≥ τ_k`; their multipliers grow when violated and
relax when satisfied. The model gets dynamic, per-skill pressure
without hand-tuning fixed weights.

## What's in this recipe

| File | Purpose |
| --- | --- |
| [`controller.py`](controller.py) | `PrimalDualController` (single-writer singleton): λ, τ, EMA stats, dual update, checkpointable state. |
| [`advantage.py`](advantage.py) | `compute_pd_gdpo_outcome_advantage` registered as `pd_gdpo`. Mirrors the GDPO estimator with correctness gating + λ-aggregation + post-pricing dual update. |
| [`custom_reward.py`](custom_reward.py) | `compute_score` that returns the primary reward as `"score"` and each auxiliary sub-reward as its own key; verl's `NaiveRewardManager` lifts these into `non_tensor_batch`, where the advantage estimator reads them. |
| [`reward_score/sub_reward/`](reward_score/sub_reward/) | Vendored heuristic sub-rewards (math + coding) from the [pd_reward branch](https://github.com/Lemutisme/verl/tree/pd_reward/recipe/skill_evo). |
| [`config/pd_gdpo_trainer.yaml`](config/pd_gdpo_trainer.yaml) | Hydra config extending `ppo_trainer.yaml` with an `algorithm.pd_gdpo.*` block. |
| [`main_pdpo.py`](main_pdpo.py) | Entry point; importing it triggers `recipe.pdpo` side-effect registration. |
| [`run_pd_gdpo_math.sh`](run_pd_gdpo_math.sh) | Minimal launcher example. |
| [`tests/`](tests/) | Smoke + unit tests for the controller and the estimator. |

The only modifications outside this recipe are:

* `verl/trainer/ppo/ray_trainer.py` — one literal added to the existing GDPO
  passthrough so `non_tensor_batch` + `batch` are forwarded to the
  `pd_gdpo` estimator. No new function, no new dispatch path.
* `verl/trainer/config/algorithm.py` — adds an `algorithm.pd_gdpo: dict`
  field so Hydra can populate it from YAML/CLI.

## Quickstart

```bash
# 1. Make sure the recipe submodule is initialized
git submodule update --init recipe

# 2. Set MODEL_PATH / TRAIN_FILE / TEST_FILE for your env, then:
bash recipe/pdpo/run_pd_gdpo_math.sh
```

Minimal CLI override of the estimator only:

```bash
python -m recipe.pdpo.main_pdpo \
    algorithm.adv_estimator=pd_gdpo \
    algorithm.pd_gdpo.component_keys='[math_final_answer_reward,math_answer_efficiency_reward,math_consistency_reward]' \
    reward.custom_reward_function.path=recipe/pdpo/custom_reward.py \
    reward.custom_reward_function.name=compute_score \
    reward.custom_reward_function.reward_kwargs='{math_enable_sub_rewards: true}' \
    data.train_files=... data.val_files=... \
    actor_rollout_ref.model.path=...
```

## Configuration

All knobs live under `algorithm.pd_gdpo`. The full list with defaults
is in [`config/pd_gdpo_trainer.yaml`](config/pd_gdpo_trainer.yaml); the
ones you actually care about:

| Key | Meaning |
| --- | --- |
| `component_keys` | Ordered list of auxiliary component names. Must match keys in your `compute_score` return dict. The primary reward is **not** listed here. |
| `correctness_gate` | Auxiliary residuals are masked out for samples with `primary <= correctness_gate`. Set to 0.0 for binary correctness; pick a meaningful floor for continuous primary rewards. |
| `perf_lo`, `perf_hi` | τ-schedule interpolation range over `ema_primary`. |
| `ema_alpha`, `warmup_alpha`, `warmup_steps` | EMA mixing rates for primary + components. Warmup applies to the first N batches. |
| `eta_gate_center`, `eta_gate_scale`, `eta_decay` | η is sigmoid-gated by `ema_primary` and (optionally) 1/√step decayed, so dual updates only start mattering once primary is non-trivially above floor. |
| `rho_mode` | `"dual_mass"` (default) or `"raw"`. `dual_mass` keeps primary advantage proportionally dominant when many constraints are violated; synced from upstream pd_reward's `normalize_by_dual_mass=True` default. |
| `dual_update` | `"additive"` (default, projected sub-gradient) or `"mirror"` (multiplicative). |
| `adv_clip` | Per-component group-advantage clamp band before λ-weighting. Set to 0 to disable. |
| `component_defaults`, `components.<name>` | Per-component `tau_min`/`tau_max`/`eta`/`lambda_max`/`lambda_init`/`monotone_tau`. `lambda_max` defaults to **2.0** (lowered from 4.0; upstream uses 0.5 for scalarized PD). |
| `eta_decay` | Log-decay schedule `η ← η / (1 + 0.1·ln(step+1))` (sync'd from upstream; replaces 1/√step which was killing updates too quickly). |
| `state_path` | Optional JSON for controller checkpointing. |

Every Hydra knob has a matching `PDGDPO_*` env-var override (consulted
last, so it wins over YAML/CLI). The full mapping is in
[`controller.py`](controller.py)'s `ControllerConfig._apply_env`.

## Behavior notes

* **Single writer.** The controller is a process-local singleton
  constructed lazily on first call. The advantage estimator is the
  **only** code path that updates λ; it does so exactly once per
  rollout batch, *after* pricing. The reward manager never sees λ.
* **Gating asymmetry.** Components are only updated for prompt batches
  with at least one gated sample. Otherwise we'd dual-step toward a
  signal that didn't actually appear in the training batch.
* **Singleton groups.** When a prompt has only one rollout, its
  per-component group advantage is zero (matches GRPO semantics).
* **Component-adv clamp.** Per-component group advantages are clamped
  to ±`adv_clip` before λ-weighting; this protects against spikes when
  only one sample in a group passes the gate.
* **τ monotonicity.** Each component supports `monotone_tau: true`,
  which prevents τ from dropping back when the primary EMA briefly
  regresses — useful when you want the model to never *unlearn* an
  auxiliary skill.

## Ablation matrix to actually run

| Tag | Reward path | Advantage |
| --- | --- | --- |
| `grpo-primary` | primary only | scalar GRPO |
| `grpo-static` | fixed scalar sum | scalar GRPO |
| `pd-scalar` | primal-dual scalarized | scalar GRPO |
| `gdpo-static` | components | GDPO + fixed weights |
| `pd-gdpo` | components | GDPO + λ |
| `pd-gdpo-nogate` | components | GDPO + λ, gate disabled |
| `pd-gdpo-λ-pre-norm` | components, λ baked in before norm | diagnostic |

The most important comparison is **pd-scalar vs pd-gdpo**: it isolates
the effect of moving primal-dual control from reward to advantage. The
other rows answer "is the gain coming from PD or just from GDPO?".

## Sync history with upstream pd_reward

This recipe vendors heuristic sub-rewards from
[Lemutisme/verl:pd_reward](https://github.com/Lemutisme/verl/tree/pd_reward/recipe/skill_evo)
and tracks its dual-update stability fixes.

| Upstream commit | Synced? | Notes |
| --- | --- | --- |
| `85bb22c5` heuristics, sub_reward layout | Vendored | `recipe/pdpo/reward_score/sub_reward/` |
| `293fa204` deterministic subreward order | N/A | PD-GDPO iterates components in config-declared order |
| `2b9e9fa2` log-eta decay | Synced | `controller.py:update_duals` uses `1/(1 + 0.1·ln(step+1))` |
| `2b9e9fa2` lambda_max = 0.5 default | Adapted | PD-GDPO default = 2.0 (whitening neutralizes scale, but smaller is safer) |
| `2b9e9fa2` normalize_by_dual_mass = True default | Synced | `rho_mode: dual_mass` is now default |
| `2b9e9fa2` math_final_answer_reward disabled by default | Synced | `reward_score/sub_reward/__init__.py:DEFAULT_ENABLED` |
| `2b9e9fa2` deferred per-step dual updates (`flush_pending_duals`, `global_step`) | **N/A by design** | PD-GDPO calls `controller.update_duals` exactly once per batch from the advantage estimator -- no step inflation possible. |
| `396de68c` data-preprocess scripts | Out of scope | These are dataset-specific shell scripts, not algorithm changes. |

## Acknowledgements

* Heuristic sub-rewards (math + coding) vendored from the
  [pd_reward branch on Lemutisme/verl](https://github.com/Lemutisme/verl/tree/pd_reward/recipe/skill_evo).
* GDPO geometry from [NVlabs/GDPO](https://github.com/NVlabs/GDPO).
