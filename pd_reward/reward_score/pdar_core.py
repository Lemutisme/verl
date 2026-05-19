"""
PDAR Core — Primal-Dual Advantage Regulation

This module implements the dual-variable state management and selective
sharpness damping logic for PDAR.  It operates on PyTorch tensors in the
advantage space (not raw rewards).

Key concepts
------------
* **Constraint dual** (`lambda_c`): controls the influence of auxiliary
  advantage signals on the combined advantage.  Updated based on whether
  the auxiliary metric meets its target.
* **Sharpness dual** (`lambda_s`): selectively dampens extreme within-group
  advantages using a *bounded influence function* (`d / (1 + λ|d|)`),
  controlling update geometry rather than merely rescaling the learning rate.

Reference positioning
---------------------
* GDPO (arXiv:2601.05242) normalises each reward channel independently with
  **fixed** weights.  PDAR extends this to **feedback-controlled** weights.
* Constrained GRPO (arXiv:2602.05863) uses scalarised Lagrangian advantage.
  PDAR uses component-wise normalised combination and explicitly does **not**
  claim formal CMDP guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PDARConfig:
    """Hyper-parameters for PDAR advantage regulation."""

    # --- Constraint dual ---
    eta_c: float = 0.05           # step size for λ_c update
    lambda_c_max: float = 1.0     # hard cap on λ_c
    tau_c: float = 0.5            # target for mean auxiliary reward
    sign_c: float = 1.0           # +1 if higher aux is better, -1 if violation

    # --- Sharpness dual ---
    eta_s: float = 0.01           # step size for λ_s update  (<<  eta_c)
    lambda_s_max: float = 2.0     # hard cap on λ_s
    tau_s: float = 1.5            # target std (sharpness threshold)
    sharpness_ema_alpha: float = 0.1  # EMA smoothing for sharpness statistic

    # --- General ---
    epsilon: float = 1e-6         # numerical stability
    norm_adv_by_std: bool = True  # whether to divide by std in GRPO

    @classmethod
    def from_dict(cls, d: dict) -> "PDARConfig":
        """Construct from a flat dictionary (e.g. from Hydra reward_kwargs).

        Keys are expected to be prefixed with ``pdar_`` — e.g.
        ``pdar_eta_c``, ``pdar_lambda_c_max``.
        """
        prefix = "pdar_"
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            short = k[len(prefix):] if k.startswith(prefix) else k
            if short in field_names:
                # coerce to expected type
                target_type = type(getattr(cls, short, v))
                try:
                    filtered[short] = target_type(v)
                except (ValueError, TypeError):
                    filtered[short] = v
        return cls(**filtered)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class PDARState:
    """Mutable state for the PDAR dual variables."""

    lambda_c: float = 0.0
    lambda_s: float = 0.0
    sharpness_ema: float = 0.0
    step: int = 0

    # optional per-signal lambdas for multi-channel extension
    lambda_per_signal: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dual variable updates
# ---------------------------------------------------------------------------

def update_constraint_dual(
    state: PDARState,
    config: PDARConfig,
    aux_mean: float,
) -> None:
    """Update the constraint dual based on how far the auxiliary metric is from its target.

    Rule:  λ_c ← clamp( λ_c + η_c · (τ_c − aux_mean), 0, λ_c_max )

    If ``sign_c = +1`` (higher aux is better), violation = target − actual > 0 when under-performing.
    If ``sign_c = -1`` (lower aux is better, e.g. violation), flip the sign.
    """
    violation = config.sign_c * (config.tau_c - aux_mean)
    state.lambda_c = max(0.0, min(
        state.lambda_c + config.eta_c * violation,
        config.lambda_c_max,
    ))


def update_sharpness_dual(
    state: PDARState,
    config: PDARConfig,
    current_sharpness: float,
) -> None:
    """Update the sharpness dual using EMA-smoothed group advantage std.

    Rule:
      S_ema ← (1 − α) · S_ema + α · S_G
      λ_s  ← clamp( λ_s + η_s · (S_ema − τ_s), 0, λ_s_max )
    """
    # EMA smooth
    if state.step == 0:
        state.sharpness_ema = current_sharpness
    else:
        alpha = config.sharpness_ema_alpha
        state.sharpness_ema = (1 - alpha) * state.sharpness_ema + alpha * current_sharpness

    excess = state.sharpness_ema - config.tau_s
    state.lambda_s = max(0.0, min(
        state.lambda_s + config.eta_s * excess,
        config.lambda_s_max,
    ))

    state.step += 1


# ---------------------------------------------------------------------------
# Selective sharpness damping
# ---------------------------------------------------------------------------

def selective_damp(
    advantages: torch.Tensor,
    lambda_s: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply bounded-influence sharpness damping.

    For each advantage value *a*, compute the normalised deviation
    ``d = (a − mean) / (std + ε)`` and apply the non-linear transform
    ``d_stable = d / (1 + λ_s · |d|)``.

    Properties:
      - Small advantages (~mean) are almost unchanged.
      - Extreme advantages are non-linearly compressed toward ±1/λ_s.
      - Sign and approximate ranking are preserved.
      - When λ_s = 0, this is the identity function.

    Args:
        advantages: 1-D tensor of advantage values for one group.
        lambda_s: current sharpness dual variable.
        eps: numerical stability constant.

    Returns:
        Damped advantages, same shape as input.
    """
    if lambda_s <= 0.0:
        return advantages

    mean = advantages.mean()
    std = advantages.std() + eps
    d = (advantages - mean) / std
    d_stable = d / (1.0 + lambda_s * d.abs())
    return mean + std * d_stable


# ---------------------------------------------------------------------------
# Group-level GRPO normalisation (utility shared with pdar_advantage.py)
# ---------------------------------------------------------------------------

def group_normalize_scores(
    scores: torch.Tensor,
    index: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    norm_by_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute group-relative normalised scores (GRPO-style).

    For each group identified by ``index``, centre scores by the group mean
    and optionally divide by the group std.

    Args:
        scores: 1-D tensor of scalar scores, shape ``(batch_size,)``.
        index: array of group identifiers, shape ``(batch_size,)``.
        norm_by_std: whether to divide by group std (True = GRPO, False = Dr.GRPO).
        eps: numerical stability constant.

    Returns:
        Normalised scores, shape ``(batch_size,)``.
    """
    import numpy as np  # local import to keep module import-light

    id2scores: dict = {}
    bsz = scores.shape[0]

    with torch.no_grad():
        for i in range(bsz):
            id2scores.setdefault(index[i], []).append(scores[i])

        id2mean: dict = {}
        id2std: dict = {}
        for idx, group_scores in id2scores.items():
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                stacked = torch.stack(group_scores)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        normed = scores.clone()
        for i in range(bsz):
            if norm_by_std:
                normed[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
            else:
                normed[i] = scores[i] - id2mean[index[i]]

    return normed


# ---------------------------------------------------------------------------
# Convenience: get / create singleton PDAR state
# ---------------------------------------------------------------------------

_PDAR_STATE: Optional[PDARState] = None
_PDAR_CONFIG: Optional[PDARConfig] = None


def get_pdar_state(config_dict: Optional[dict] = None) -> tuple[PDARState, PDARConfig]:
    """Return the module-level singleton ``(state, config)`` pair.

    On first call, initialises from *config_dict*.  Subsequent calls
    return the same objects (dual variables persist across training steps).
    """
    global _PDAR_STATE, _PDAR_CONFIG
    if _PDAR_STATE is None:
        _PDAR_CONFIG = PDARConfig.from_dict(config_dict or {})
        _PDAR_STATE = PDARState()
    return _PDAR_STATE, _PDAR_CONFIG


def reset_pdar_state() -> None:
    """Reset the singleton PDAR state (useful for testing)."""
    global _PDAR_STATE, _PDAR_CONFIG
    _PDAR_STATE = None
    _PDAR_CONFIG = None
