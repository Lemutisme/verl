"""
PDPO Advantage Estimator.

PDPO keeps the original task reward as the anchor, but uses process-distance
subrewards to estimate advantage when the original reward is uninformative
inside a GRPO group.  Unlike scalar reward mixing, each subreward channel is
group-normalised independently before aggregation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est

from reward_score.pdar_core import group_normalize_scores, selective_damp


PDPO_METRICS: dict[str, float] = {}


@dataclass
class PDPOConfig:
    """Hyper-parameters for PDPO advantage estimation."""

    beta_tie: float = 0.20
    beta_same: float = 1.00
    lambda_aux: float = 1.00
    min_aux_std: float = 1e-6
    min_main_std: float = 1e-6

    eta_s: float = 0.01
    lambda_s_max: float = 2.0
    tau_s: float = 1.5
    sharpness_ema_alpha: float = 0.1

    epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PDPOConfig":
        aliases = {
            "beta_wrong": "beta_same",
            "beta_flat": "beta_same",
            "beta_no_main": "beta_same",
        }
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered: dict[str, Any] = {}

        for key, value in d.items():
            candidates = [key]
            if key.startswith("pdpo_"):
                candidates.append(key[len("pdpo_"):])
            if key.startswith("pdar_"):
                candidates.append(key[len("pdar_"):])

            for candidate in candidates:
                short = aliases.get(candidate, candidate)
                if short not in field_names:
                    continue
                target_type = type(getattr(cls, short))
                try:
                    filtered[short] = target_type(value)
                except (TypeError, ValueError):
                    filtered[short] = value

        return cls(**filtered)


@dataclass
class PDPOState:
    lambda_s: float = 0.0
    sharpness_ema: float = 0.0
    step: int = 0


_PDPO_STATE: Optional[PDPOState] = None
_PDPO_CONFIG: Optional[PDPOConfig] = None


def get_pdpo_state(config_dict: Optional[dict[str, Any]] = None) -> tuple[PDPOState, PDPOConfig]:
    global _PDPO_STATE, _PDPO_CONFIG
    if _PDPO_STATE is None:
        _PDPO_CONFIG = PDPOConfig.from_dict(config_dict or {})
        _PDPO_STATE = PDPOState()
    return _PDPO_STATE, _PDPO_CONFIG


def reset_pdpo_state() -> None:
    global _PDPO_STATE, _PDPO_CONFIG
    _PDPO_STATE = None
    _PDPO_CONFIG = None


def _plain_config_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if is_dataclass(value):
        return asdict(value)
    return dict(value) if isinstance(value, dict) else {}


def _to_scalar_tensor(values: Any, *, bsz: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(device=device, dtype=dtype)
        if tensor.ndim > 1:
            tensor = tensor.sum(dim=tuple(range(1, tensor.ndim)))
        tensor = tensor.reshape(-1)
        return tensor if tensor.numel() == bsz else None

    try:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != bsz:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(arr, device=device, dtype=dtype)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _channel_weight(name: str, config_dict: dict[str, Any]) -> float:
    candidates = [f"weight_{name}"]
    if name.startswith("math_"):
        candidates.append(f"math_weight_{name[len('math_'):]}")
    if name.startswith("coding_"):
        candidates.append(f"coding_weight_{name[len('coding_'):]}")
    if name in {"thought", "action"}:
        candidates.append(f"weight_{name}")

    for candidate in candidates:
        if candidate in config_dict:
            return _to_float(config_dict[candidate], 1.0)
    return 1.0


_EXCLUDED_AUX_KEYS = {
    "acc",
    "any_pass",
    "aux_reward_combined",
    "aux_rewards",
    "base_math_score",
    "combined_reward",
    "main_reward",
    "original_reward",
    "partial_pass_rate",
    "reward",
    "score",
}


def _is_aux_channel(name: str) -> bool:
    return name not in _EXCLUDED_AUX_KEYS and (
        name.startswith("math_")
        or name.startswith("coding_")
        or name in {"thought", "action"}
    )


def _group_indices(index: np.ndarray, bsz: int) -> dict[Any, list[int]]:
    id2indices: dict[Any, list[int]] = defaultdict(list)
    for i in range(bsz):
        id2indices[index[i]].append(i)
    return id2indices


def _active_group_mask(scores: torch.Tensor, id2indices: dict[Any, list[int]], min_std: float) -> torch.Tensor:
    mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    for idxs in id2indices.values():
        if len(idxs) <= 1:
            continue
        idx_tensor = torch.tensor(idxs, device=scores.device)
        if scores[idx_tensor].std().item() > min_std:
            mask[idx_tensor] = True
    return mask


def _update_sharpness_dual(state: PDPOState, cfg: PDPOConfig, current_sharpness: float) -> None:
    if state.step == 0:
        state.sharpness_ema = current_sharpness
    else:
        alpha = cfg.sharpness_ema_alpha
        state.sharpness_ema = (1.0 - alpha) * state.sharpness_ema + alpha * current_sharpness

    state.lambda_s = max(
        0.0,
        min(state.lambda_s + cfg.eta_s * (state.sharpness_ema - cfg.tau_s), cfg.lambda_s_max),
    )
    state.step += 1


@register_adv_est("pdpo")
def compute_pdpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
    pdpo_aux_rewards_dict: Optional[dict[str, Any]] = None,
    aux_rewards_tensor: Optional[torch.Tensor] = None,
    pdpo_config_dict: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PDPO advantages.

    PDPO uses ``A_main`` from the original reward and adds independently
    normalised auxiliary advantages.  Auxiliary channels get full weight only
    in groups where the original reward has no variance; otherwise they are a
    small tie-breaker.
    """
    global PDPO_METRICS

    bsz = token_level_rewards.shape[0]
    if index is None:
        index = np.arange(bsz, dtype=object)

    config_dict = _plain_config_dict(config)
    config_dict.update(pdpo_config_dict or {})
    state, cfg = get_pdpo_state(config_dict)
    eps = _to_float(config_dict.get("pdpo_epsilon", cfg.epsilon), epsilon)

    main_scores = token_level_rewards.sum(dim=-1)
    id2indices = _group_indices(index, bsz)
    main_adv = group_normalize_scores(
        main_scores,
        index,
        norm_by_std=norm_adv_by_std_in_grpo,
        eps=eps,
    )

    aux_sources: dict[str, Any] = dict(pdpo_aux_rewards_dict or {})
    if not aux_sources and aux_rewards_tensor is not None:
        aux_sources["aux_tensor"] = aux_rewards_tensor

    aux_adv_sum = torch.zeros_like(main_adv)
    aux_raw_sum = torch.zeros_like(main_adv)
    active_channels = 0
    active_group_count = 0
    weight_sum = 0.0
    channel_metrics: dict[str, float] = {}

    for name in sorted(aux_sources):
        if name != "aux_tensor" and not _is_aux_channel(name):
            continue
        weight = _channel_weight(name, config_dict)
        if weight <= 0.0:
            continue

        aux_scores = _to_scalar_tensor(
            aux_sources[name],
            bsz=bsz,
            device=token_level_rewards.device,
            dtype=token_level_rewards.dtype,
        )
        if aux_scores is None:
            continue

        active_mask = _active_group_mask(aux_scores, id2indices, cfg.min_aux_std)
        if not active_mask.any().item():
            continue

        aux_adv = group_normalize_scores(
            aux_scores,
            index,
            norm_by_std=norm_adv_by_std_in_grpo,
            eps=eps,
        )
        aux_adv = torch.where(active_mask, aux_adv, torch.zeros_like(aux_adv))
        aux_adv_sum = aux_adv_sum + weight * aux_adv
        aux_raw_sum = aux_raw_sum + weight * aux_scores
        active_channels += 1
        weight_sum += weight

        active_group_count += sum(
            1 for idxs in id2indices.values()
            if len(idxs) > 1 and aux_scores[torch.tensor(idxs, device=aux_scores.device)].std().item() > cfg.min_aux_std
        )
        metric_name = name.replace("/", "_")
        channel_metrics[f"pdpo/channel/{metric_name}/weight"] = float(weight)
        channel_metrics[f"pdpo/channel/{metric_name}/mean"] = float(aux_scores.mean().item())

    main_active_mask = _active_group_mask(main_scores, id2indices, cfg.min_main_std)
    beta = torch.where(
        main_active_mask,
        torch.full_like(main_adv, cfg.beta_tie),
        torch.full_like(main_adv, cfg.beta_same),
    )
    combined_adv = main_adv + cfg.lambda_aux * beta * aux_adv_sum

    group_stds_before: list[float] = []
    group_stds_after: list[float] = []
    damped_adv = combined_adv.clone()
    for idxs in id2indices.values():
        if len(idxs) <= 1:
            continue
        idx_tensor = torch.tensor(idxs, device=combined_adv.device)
        group_adv = combined_adv[idx_tensor]
        group_stds_before.append(group_adv.std().item())
        damped_group = selective_damp(group_adv, state.lambda_s, eps=eps)
        damped_adv[idx_tensor] = damped_group
        group_stds_after.append(damped_group.std().item())

    mean_sharpness = sum(group_stds_before) / len(group_stds_before) if group_stds_before else 0.0
    _update_sharpness_dual(state, cfg, mean_sharpness)

    advantages = damped_adv.unsqueeze(-1) * response_mask
    mean_std_after = sum(group_stds_after) / len(group_stds_after) if group_stds_after else 0.0
    PDPO_METRICS = {
        "pdpo/lambda_s": state.lambda_s,
        "pdpo/sharpness_ema": state.sharpness_ema,
        "pdpo/group_adv_std_before": mean_sharpness,
        "pdpo/group_adv_std_after": mean_std_after,
        "pdpo/main_reward_mean": float(main_scores.mean().item()),
        "pdpo/aux_mean": float(aux_raw_sum.mean().item()) if active_channels else 0.0,
        "pdpo/active_channels": float(active_channels),
        "pdpo/active_group_count": float(active_group_count),
        "pdpo/weight_sum": float(weight_sum),
        "pdpo/beta_tie": cfg.beta_tie,
        "pdpo/beta_same": cfg.beta_same,
        "pdpo/lambda_aux": cfg.lambda_aux,
        "pdpo/step": float(state.step),
    }
    PDPO_METRICS.update(channel_metrics)

    return advantages, advantages
