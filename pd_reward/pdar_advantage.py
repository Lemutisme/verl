"""
PDAR Advantage Estimator — Primal-Dual Advantage Regulation for GRPO.

Registered with verl's advantage-estimator registry under the name ``"pdar"``.
When ``compute_advantage()`` in ``ray_trainer.py`` dispatches to this estimator,
it performs:

1. **Decoupled normalisation** (GDPO-style): group-normalise main and aux
   reward channels independently.
2. **Dual-controlled combination**: ``Ã = A_main + sign · λ_c · A_aux``.
3. **Selective sharpness damping**: non-linearly compress extreme within-group
   advantages using a bounded influence function.
4. **Dual variable update**: adjust ``λ_c`` and ``λ_s`` based on batch
   statistics.

The estimator degrades to vanilla GRPO when both dual variables are zero.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.config import AlgoConfig

from reward_score.pdar_core import (
    PDARConfig,
    PDARState,
    get_pdar_state,
    group_normalize_scores,
    selective_damp,
    update_constraint_dual,
    update_sharpness_dual,
)


# Module-level storage for metrics from the last call.
# The training loop can read this after compute_advantage() returns.
PDAR_METRICS: dict[str, float] = {}


@register_adv_est("pdar")
def compute_pdar_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    # PDAR-specific: aux reward tensor (same shape as token_level_rewards)
    aux_rewards_tensor: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PDAR-regulated group-relative advantages.

    Args:
        token_level_rewards: ``(bs, response_length)`` — main reward signal.
        response_mask: ``(bs, response_length)`` — 1 for valid tokens, 0 for padding.
        index: ``(bs,)`` — group IDs (uid) for GRPO grouping.
        epsilon: small constant for numerical stability.
        norm_adv_by_std_in_grpo: whether to divide by group std (GRPO) or not (Dr.GRPO).
        config: algorithm config (may contain PDAR overrides).
        aux_rewards_tensor: ``(bs, response_length)`` — auxiliary reward signal.
            If ``None``, fall back to vanilla GRPO (no dual regulation).

    Returns:
        ``(advantages, returns)`` — both of shape ``(bs, response_length)``.
    """
    global PDAR_METRICS

    # ----- 1. Extract scalar scores per response -----
    main_scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # ----- 2. Get PDAR config and state -----
    config_dict = {}
    if config is not None:
        # Try to extract pdar_* keys from config (OmegaConf or dict)
        try:
            config_dict = dict(config)
        except Exception:
            config_dict = {}
    state, pdar_cfg = get_pdar_state(config_dict)

    # ----- 3. Normalise main scores (GRPO-style, group-relative) -----
    main_adv = group_normalize_scores(
        main_scores, index,
        norm_by_std=norm_adv_by_std_in_grpo,
        eps=epsilon,
    )  # (bs,)

    # ----- 4. Handle aux channel (if available) -----
    if aux_rewards_tensor is not None:
        aux_scores = aux_rewards_tensor.sum(dim=-1)  # (bs,)
        aux_adv = group_normalize_scores(
            aux_scores, index,
            norm_by_std=norm_adv_by_std_in_grpo,
            eps=epsilon,
        )  # (bs,)

        # Combined advantage: Ã = A_main + sign · λ_c · A_aux
        combined_adv = main_adv + pdar_cfg.sign_c * state.lambda_c * aux_adv
    else:
        # No aux signal → fall back to vanilla GRPO
        combined_adv = main_adv
        aux_scores = None

    # ----- 5. Per-group selective sharpness damping -----
    # Collect group-level sharpness statistics before damping
    id2indices: dict[Any, list[int]] = defaultdict(list)
    bsz = combined_adv.shape[0]
    for i in range(bsz):
        id2indices[index[i]].append(i)

    # Compute mean group std (sharpness metric)
    group_stds_before: list[float] = []
    group_stds_after: list[float] = []

    damped_adv = combined_adv.clone()
    for gid, idxs in id2indices.items():
        if len(idxs) <= 1:
            continue
        group_advs = combined_adv[idxs]
        group_std = group_advs.std().item()
        group_stds_before.append(group_std)

        # Apply selective damping per group
        damped_group = selective_damp(group_advs, state.lambda_s, eps=epsilon)
        for j, idx in enumerate(idxs):
            damped_adv[idx] = damped_group[j]

        group_stds_after.append(damped_group.std().item())

    # ----- 6. Update dual variables -----
    # Sharpness: mean group std
    if group_stds_before:
        mean_sharpness = sum(group_stds_before) / len(group_stds_before)
    else:
        mean_sharpness = 0.0
    update_sharpness_dual(state, pdar_cfg, mean_sharpness)

    # Constraint: mean aux reward (raw, not normalised)
    if aux_scores is not None:
        aux_mean = aux_scores.mean().item()
        update_constraint_dual(state, pdar_cfg, aux_mean)
    else:
        aux_mean = 0.0

    # ----- 7. Broadcast to token level -----
    advantages = damped_adv.unsqueeze(-1) * response_mask  # (bs, resp_len)

    # ----- 8. Record metrics -----
    mean_std_after = sum(group_stds_after) / len(group_stds_after) if group_stds_after else 0.0
    PDAR_METRICS = {
        "pdar/lambda_c": state.lambda_c,
        "pdar/lambda_s": state.lambda_s,
        "pdar/sharpness_ema": state.sharpness_ema,
        "pdar/group_adv_std_before": mean_sharpness,
        "pdar/group_adv_std_after": mean_std_after,
        "pdar/aux_mean": aux_mean,
        "pdar/main_reward_mean": main_scores.mean().item(),
        "pdar/step": float(state.step),
    }

    return advantages, advantages
