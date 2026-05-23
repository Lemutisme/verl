"""
PDPO Advantage Estimator.

PDPO keeps the original task reward as the anchor, but uses process-distance
subrewards to estimate advantage when the original reward is uninformative
inside a GRPO group.  Unlike scalar reward mixing, each subreward channel is
group-normalised independently before aggregation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


PDPO_METRICS: dict[str, float] = {}


@dataclass
class PDPOConfig:
    """Hyper-parameters for PDPO advantage estimation."""

    beta_tie: float = 0.0
    beta_same: float = 0.70
    lambda_aux: float = 0.70
    lambda_aux_start: float = 0.30
    lambda_aux_warmup_steps: int = 0
    min_aux_std: float = 1e-6
    min_main_std: float = 1e-6
    answer_gate_channel: str = "math_answer_extractability_reward"
    answer_gate_min: float = 0.5
    answer_gate_closed_scale: float = 0.0
    answer_gate_as_constraint: bool = True
    answer_gate_preference_scale: float = 0.0

    correctness_safe: bool = True
    correctness_margin: float = 1e-3

    reliability_enabled: bool = True
    reliability_ema_alpha: float = 0.05
    reliability_min_scale: float = 0.0
    reliability_max_scale: float = 1.0
    reliability_target_margin: float = 0.02
    reliability_negative_tolerance: float = 0.02
    reliability_wrong_high_threshold: float = 0.30
    reliability_wrong_high_target: float = 0.20
    reliability_pairwise_target: float = 0.55
    reliability_inversion_target: float = 0.20
    reliability_min_comparable_groups: int = 1
    reliability_wrong_high_smoothing: float = 0.0

    safety_dual_enabled: bool = True
    safety_dual_eta: float = 0.05
    safety_dual_mu_max: float = 6.0
    safety_dual_decay: float = 0.0
    safety_dual_target_margin: float = 0.02
    safety_dual_wrong_high_target: float = 0.20
    safety_dual_inversion_target: float = 0.20
    safety_dual_min_comparable_groups: int = 1
    safety_dual_ema_alpha: float = 0.10
    safety_dual_recovery_scale: float = 0.25

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

            for candidate in candidates:
                short = aliases.get(candidate, candidate)
                if short not in field_names:
                    continue
                filtered[short] = _coerce_config_value(value, getattr(cls, short))

        return cls(**filtered)


@dataclass
class PDPOState:
    lambda_s: float = 0.0
    sharpness_ema: float = 0.0
    step: int = 0
    channel_reliability: dict[str, float] = field(default_factory=dict)
    channel_safety_dual: dict[str, float] = field(default_factory=dict)
    channel_safety_pressure_ema: dict[str, float] = field(default_factory=dict)


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


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if value is None:
        return default
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


def _coerce_config_value(value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return _to_bool(value, default_value)
    target_type = type(default_value)
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _channel_weight(name: str, config_dict: dict[str, Any]) -> float:
    candidates = [f"weight_{name}"]
    if name.startswith("math_"):
        candidates.append(f"math_weight_{name[len('math_'):]}")
    if name.startswith("coding_"):
        candidates.append(f"coding_weight_{name[len('coding_'):]}")
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
    )


def _group_indices(index: np.ndarray, bsz: int) -> dict[Any, list[int]]:
    id2indices: dict[Any, list[int]] = defaultdict(list)
    for i in range(bsz):
        id2indices[index[i]].append(i)
    return id2indices


def group_normalize_scores(
    scores: torch.Tensor,
    index: np.ndarray,
    norm_by_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute group-relative scores for one scalar channel."""
    id2scores: dict[Any, list[torch.Tensor]] = defaultdict(list)
    bsz = scores.shape[0]

    with torch.no_grad():
        for i in range(bsz):
            id2scores[index[i]].append(scores[i])

        id2mean: dict[Any, torch.Tensor] = {}
        id2std: dict[Any, torch.Tensor] = {}
        for group_id, group_scores in id2scores.items():
            if len(group_scores) == 1:
                id2mean[group_id] = torch.tensor(0.0, device=scores.device)
                id2std[group_id] = torch.tensor(1.0, device=scores.device)
            else:
                stacked = torch.stack(group_scores)
                id2mean[group_id] = stacked.mean()
                id2std[group_id] = stacked.std()

        normed = scores.clone()
        for i in range(bsz):
            if norm_by_std:
                normed[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
            else:
                normed[i] = scores[i] - id2mean[index[i]]
    return normed


def selective_damp(
    advantages: torch.Tensor,
    lambda_s: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compress extreme within-group advantages while preserving rank and sign."""
    if lambda_s <= 0.0:
        return advantages

    mean = advantages.mean()
    std = advantages.std() + eps
    deviation = (advantages - mean) / std
    stable_deviation = deviation / (1.0 + lambda_s * deviation.abs())
    return mean + std * stable_deviation


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


def _channel_reliability_stats(
    aux_scores: torch.Tensor,
    main_scores: torch.Tensor,
    id2indices: dict[Any, list[int]],
    cfg: PDPOConfig,
    eps: float,
) -> dict[str, float]:
    correct_chunks: list[torch.Tensor] = []
    wrong_chunks: list[torch.Tensor] = []
    supporting_groups = 0
    comparable_groups = 0
    pairwise_total = 0.0
    pairwise_aligned = 0.0
    pairwise_inverted = 0.0
    pairwise_tied = 0.0

    for idxs in id2indices.values():
        if len(idxs) <= 1:
            continue
        idx_tensor = torch.tensor(idxs, device=main_scores.device)
        group_main = main_scores[idx_tensor]
        if group_main.std().item() <= cfg.min_main_std:
            continue

        best_main = group_main.max()
        correct_mask = torch.isclose(group_main, best_main, atol=eps, rtol=0.0)
        wrong_mask = ~correct_mask
        if not correct_mask.any().item() or not wrong_mask.any().item():
            continue

        group_aux = aux_scores[idx_tensor]
        higher_main = group_main[:, None] > group_main[None, :] + eps
        if higher_main.any().item():
            higher_aux = group_aux[:, None]
            lower_aux = group_aux[None, :]
            aligned = higher_main & (higher_aux > lower_aux + eps)
            inverted = higher_main & (higher_aux + eps < lower_aux)
            tied = higher_main & ~(aligned | inverted)
            pairwise_total += float(higher_main.float().sum().item())
            pairwise_aligned += float(aligned.float().sum().item())
            pairwise_inverted += float(inverted.float().sum().item())
            pairwise_tied += float(tied.float().sum().item())

        correct_aux = group_aux[correct_mask]
        wrong_aux = group_aux[wrong_mask]
        correct_chunks.append(correct_aux)
        wrong_chunks.append(wrong_aux)
        comparable_groups += 1
        if correct_aux.mean().item() + eps >= wrong_aux.mean().item():
            supporting_groups += 1

    if not comparable_groups:
        return {
            "comparable_groups": 0.0,
            "correct_mean": 0.0,
            "wrong_mean": 0.0,
            "correct_minus_wrong": 0.0,
            "wrong_high_rate": 0.0,
            "supports_correct_rate": 0.0,
            "pairwise_alignment_rate": 0.0,
            "pairwise_inversion_rate": 0.0,
            "pairwise_tie_rate": 0.0,
            "pairwise_total": 0.0,
            "batch_reliability": 1.0,
        }

    correct_values = torch.cat(correct_chunks)
    wrong_values = torch.cat(wrong_chunks)
    correct_mean = float(correct_values.mean().item())
    wrong_mean = float(wrong_values.mean().item())
    gap = correct_mean - wrong_mean
    wrong_high_count = float((wrong_values >= cfg.reliability_wrong_high_threshold).float().sum().item())
    wrong_count = float(wrong_values.numel())
    wrong_high_target = _clamp(cfg.reliability_wrong_high_target, 0.0, 1.0)
    smoothing = max(0.0, cfg.reliability_wrong_high_smoothing)
    if smoothing > 0.0:
        wrong_high_rate = (wrong_high_count + smoothing * wrong_high_target) / (wrong_count + smoothing)
    else:
        wrong_high_rate = wrong_high_count / max(wrong_count, eps)
    supports_correct_rate = supporting_groups / comparable_groups

    target_margin = max(cfg.reliability_target_margin, eps)
    gap_scale = _clamp((gap + cfg.reliability_negative_tolerance) / target_margin, 0.0, 1.0)
    if wrong_high_target >= 1.0:
        wrong_high_penalty = 1.0
    else:
        excess_high = max(0.0, wrong_high_rate - wrong_high_target)
        wrong_high_penalty = _clamp(1.0 - excess_high / max(1.0 - wrong_high_target, eps), 0.0, 1.0)
    pairwise_alignment_rate = (pairwise_aligned + 0.5 * pairwise_tied) / max(pairwise_total, eps)
    pairwise_inversion_rate = pairwise_inverted / max(pairwise_total, eps)
    pairwise_tie_rate = pairwise_tied / max(pairwise_total, eps)
    alignment_floor = 0.5
    alignment_target = max(cfg.reliability_pairwise_target, alignment_floor + eps)
    pairwise_scale = _clamp(
        (pairwise_alignment_rate - alignment_floor) / max(alignment_target - alignment_floor, eps),
        0.0,
        1.0,
    )
    inversion_target = _clamp(cfg.reliability_inversion_target, 0.0, 1.0)
    if inversion_target >= 1.0:
        inversion_penalty = 1.0
    else:
        inversion_excess = max(0.0, pairwise_inversion_rate - inversion_target)
        inversion_penalty = _clamp(1.0 - inversion_excess / max(1.0 - inversion_target, eps), 0.0, 1.0)
    batch_reliability = _clamp(pairwise_scale * inversion_penalty * wrong_high_penalty, 0.0, 1.0)

    return {
        "comparable_groups": float(comparable_groups),
        "correct_mean": correct_mean,
        "wrong_mean": wrong_mean,
        "correct_minus_wrong": gap,
        "wrong_high_rate": wrong_high_rate,
        "supports_correct_rate": float(supports_correct_rate),
        "pairwise_alignment_rate": float(pairwise_alignment_rate),
        "pairwise_inversion_rate": float(pairwise_inversion_rate),
        "pairwise_tie_rate": float(pairwise_tie_rate),
        "pairwise_total": float(pairwise_total),
        "gap_reliability": float(gap_scale),
        "wrong_high_reliability": float(wrong_high_penalty),
        "pairwise_reliability": float(pairwise_scale),
        "inversion_reliability": float(inversion_penalty),
        "batch_reliability": batch_reliability,
    }


def _effective_channel_weight(
    state: PDPOState,
    cfg: PDPOConfig,
    name: str,
    base_weight: float,
    stats: dict[str, float],
) -> tuple[float, float, float]:
    old_reliability = state.channel_reliability.get(name, 1.0)
    reliability = old_reliability
    min_groups = max(1, int(cfg.reliability_min_comparable_groups))
    should_update = stats["comparable_groups"] >= float(min_groups)
    if cfg.reliability_enabled and should_update:
        alpha = _clamp(cfg.reliability_ema_alpha, 0.0, 1.0)
        reliability = (1.0 - alpha) * old_reliability + alpha * stats["batch_reliability"]
        reliability = _clamp(reliability, cfg.reliability_min_scale, cfg.reliability_max_scale)
        state.channel_reliability[name] = reliability
    elif name not in state.channel_reliability:
        state.channel_reliability[name] = reliability

    scale = reliability if cfg.reliability_enabled else 1.0
    return base_weight * scale, reliability, float(should_update)


def _safety_dual_constraint_signal(stats: dict[str, float], cfg: PDPOConfig, eps: float) -> tuple[float, float]:
    if stats["comparable_groups"] <= 0.0:
        return 0.0, 0.0

    target_margin = max(0.0, cfg.safety_dual_target_margin)
    margin_pressure = target_margin - stats["correct_minus_wrong"]

    wrong_high_target = _clamp(cfg.safety_dual_wrong_high_target, 0.0, 1.0)
    if wrong_high_target < 1.0:
        wrong_high_pressure = (stats["wrong_high_rate"] - wrong_high_target) / max(1.0 - wrong_high_target, eps)
    else:
        wrong_high_pressure = 0.0
    inversion_target = _clamp(cfg.safety_dual_inversion_target, 0.0, 1.0)
    if inversion_target < 1.0:
        inversion_pressure = (stats.get("pairwise_inversion_rate", 0.0) - inversion_target) / max(
            1.0 - inversion_target,
            eps,
        )
    else:
        inversion_pressure = 0.0

    positive_violation = max(0.0, margin_pressure) + max(0.0, inversion_pressure) + max(0.0, wrong_high_pressure)
    if positive_violation > 0.0:
        pressure = positive_violation
    else:
        pressure = max(margin_pressure, inversion_pressure, wrong_high_pressure)

    return float(positive_violation), float(pressure)


def _update_safety_dual(
    state: PDPOState,
    cfg: PDPOConfig,
    name: str,
    stats: dict[str, float],
    eps: float,
) -> tuple[float, float, float, float, float, float]:
    if not cfg.safety_dual_enabled:
        return 0.0, 1.0, 0.0, 0.0, 0.0, 0.0

    old_mu = state.channel_safety_dual.get(name, 0.0)
    violation, pressure = _safety_dual_constraint_signal(stats, cfg, eps)
    old_pressure_ema = state.channel_safety_pressure_ema.get(name, 0.0)
    pressure_ema = old_pressure_ema
    mu = old_mu
    min_groups = max(1, int(cfg.safety_dual_min_comparable_groups))
    should_update = stats["comparable_groups"] >= float(min_groups)

    if should_update:
        alpha = _clamp(cfg.safety_dual_ema_alpha, 0.0, 1.0)
        if name in state.channel_safety_pressure_ema:
            pressure_ema = (1.0 - alpha) * old_pressure_ema + alpha * pressure
        else:
            pressure_ema = pressure
        state.channel_safety_pressure_ema[name] = pressure_ema

        decay = _clamp(cfg.safety_dual_decay, 0.0, 1.0)
        mu = max(0.0, old_mu * (1.0 - decay))
        eta = max(0.0, cfg.safety_dual_eta)
        if pressure_ema >= 0.0:
            mu = mu + eta * pressure_ema
        else:
            mu = mu + eta * max(0.0, cfg.safety_dual_recovery_scale) * pressure_ema
        mu = _clamp(mu, 0.0, max(0.0, cfg.safety_dual_mu_max))
        state.channel_safety_dual[name] = mu
    elif name not in state.channel_safety_dual:
        state.channel_safety_dual[name] = mu
        state.channel_safety_pressure_ema[name] = pressure_ema

    return float(mu), float(math.exp(-mu)), float(violation), float(pressure), float(pressure_ema), float(should_update)


def _combine_correctness_safe(
    main_adv: torch.Tensor,
    aux_component: torch.Tensor,
    main_scores: torch.Tensor,
    id2indices: dict[Any, list[int]],
    cfg: PDPOConfig,
    eps: float,
) -> tuple[torch.Tensor, int, float]:
    combined = main_adv + aux_component
    if not cfg.correctness_safe:
        return combined, 0, 0.0

    safe = combined.clone()
    clamp_count = 0
    margins: list[float] = []
    margin = max(0.0, cfg.correctness_margin)

    for idxs in id2indices.values():
        if len(idxs) <= 1:
            continue
        idx_tensor = torch.tensor(idxs, device=main_scores.device)
        group_main = main_scores[idx_tensor]
        if group_main.std().item() <= cfg.min_main_std:
            continue

        unique_scores = torch.unique(group_main.detach(), sorted=True)
        if unique_scores.numel() <= 1:
            continue

        group_main_adv = main_adv[idx_tensor]
        group_aux = aux_component[idx_tensor]
        bucket_aux = torch.zeros_like(group_aux)
        bucket_masks: list[torch.Tensor] = []
        for score in unique_scores:
            bucket_mask = torch.isclose(group_main, score, atol=eps, rtol=0.0)
            bucket_masks.append(bucket_mask)
            values = group_aux[bucket_mask]
            bucket_aux[bucket_mask] = values - values.mean()

        scale = 1.0
        for lower_mask, upper_mask in zip(bucket_masks, bucket_masks[1:]):
            lower_base = group_main_adv[lower_mask].mean().item()
            upper_base = group_main_adv[upper_mask].mean().item()
            base_gap = upper_base - lower_base
            overlap = bucket_aux[lower_mask].max().item() - bucket_aux[upper_mask].min().item()
            allowed = base_gap - margin
            if allowed <= 0.0:
                local_scale = 0.0
            elif overlap > allowed:
                local_scale = allowed / (overlap + eps)
            else:
                local_scale = 1.0
            scale = min(scale, _clamp(local_scale, 0.0, 1.0))

        if scale < 1.0 - 1e-6:
            clamp_count += 1

        group_safe = group_main_adv + bucket_aux * scale
        for lower_mask, upper_mask in zip(bucket_masks, bucket_masks[1:]):
            lower_max = group_safe[lower_mask].max().item()
            upper_min = group_safe[upper_mask].min().item()
            margins.append(upper_min - lower_max)
        safe[idx_tensor] = group_safe

    min_margin = min(margins) if margins else 0.0
    return safe, clamp_count, min_margin


def _effective_lambda_aux(state: PDPOState, cfg: PDPOConfig) -> float:
    warmup_steps = max(0, int(cfg.lambda_aux_warmup_steps))
    target = max(0.0, cfg.lambda_aux)
    if warmup_steps <= 0:
        return target

    start = _clamp(cfg.lambda_aux_start, 0.0, target)
    progress = _clamp(state.step / max(float(warmup_steps), 1.0), 0.0, 1.0)
    return start + (target - start) * progress


def _lexicographic_aux_component(
    aux_adv_sum: torch.Tensor,
    main_scores: torch.Tensor,
    id2indices: dict[Any, list[int]],
    cfg: PDPOConfig,
    eps: float,
) -> torch.Tensor:
    component = torch.zeros_like(aux_adv_sum)

    for idxs in id2indices.values():
        if len(idxs) <= 1:
            continue
        idx_tensor = torch.tensor(idxs, device=aux_adv_sum.device)
        group_main = main_scores[idx_tensor]
        group_aux = aux_adv_sum[idx_tensor]

        if group_main.std().item() <= cfg.min_main_std:
            component[idx_tensor] = cfg.beta_same * group_aux
            continue

        if cfg.beta_tie <= 0.0:
            continue

        unique_scores = torch.unique(group_main.detach(), sorted=True)
        group_component = torch.zeros_like(group_aux)
        for score in unique_scores:
            bucket_mask = torch.isclose(group_main, score, atol=eps, rtol=0.0)
            if bucket_mask.sum().item() <= 1:
                continue
            bucket_aux = group_aux[bucket_mask]
            group_component[bucket_mask] = cfg.beta_tie * (bucket_aux - bucket_aux.mean())
        component[idx_tensor] = group_component

    return component


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

    gate_name = str(config_dict.get("pdpo_answer_gate_channel", cfg.answer_gate_channel))
    gate_scores = None
    if gate_name in aux_sources:
        gate_scores = _to_scalar_tensor(
            aux_sources[gate_name],
            bsz=bsz,
            device=token_level_rewards.device,
            dtype=token_level_rewards.dtype,
        )

    aux_adv_sum = torch.zeros_like(main_adv)
    aux_raw_sum = torch.zeros_like(main_adv)
    active_channels = 0
    active_group_count = 0
    weight_sum = 0.0
    channel_metrics: dict[str, float] = {}

    for name in sorted(aux_sources):
        if name != "aux_tensor" and not _is_aux_channel(name):
            continue
        base_weight = _channel_weight(name, config_dict)
        if base_weight <= 0.0:
            continue

        aux_scores = _to_scalar_tensor(
            aux_sources[name],
            bsz=bsz,
            device=token_level_rewards.device,
            dtype=token_level_rewards.dtype,
        )
        if aux_scores is None:
            continue

        if gate_scores is not None and name != gate_name:
            gate_mask = gate_scores >= cfg.answer_gate_min
            closed_scale = max(0.0, min(1.0, cfg.answer_gate_closed_scale))
            aux_scores = torch.where(gate_mask, aux_scores, aux_scores * closed_scale)

        reliability_stats = _channel_reliability_stats(aux_scores, main_scores, id2indices, cfg, eps)
        reliability_weight, reliability, reliability_updated = _effective_channel_weight(
            state,
            cfg,
            name,
            base_weight,
            reliability_stats,
        )
        (
            safety_dual_mu,
            safety_dual_scale,
            safety_dual_violation,
            safety_dual_pressure,
            safety_dual_pressure_ema,
            safety_dual_updated,
        ) = _update_safety_dual(
            state,
            cfg,
            name,
            reliability_stats,
            eps,
        )
        effective_weight = reliability_weight * safety_dual_scale
        preference_weight = effective_weight
        if name == gate_name and cfg.answer_gate_as_constraint:
            preference_weight *= _clamp(cfg.answer_gate_preference_scale, 0.0, 1.0)
        metric_name = name.replace("/", "_")
        channel_metrics[f"pdpo/channel/{metric_name}/weight"] = float(base_weight)
        channel_metrics[f"pdpo/channel/{metric_name}/reliability_weight"] = float(reliability_weight)
        channel_metrics[f"pdpo/channel/{metric_name}/effective_weight"] = float(effective_weight)
        channel_metrics[f"pdpo/channel/{metric_name}/preference_weight"] = float(preference_weight)
        channel_metrics[f"pdpo/channel/{metric_name}/reliability"] = float(reliability)
        channel_metrics[f"pdpo/channel/{metric_name}/reliability_updated"] = float(reliability_updated)
        channel_metrics[f"pdpo/channel/{metric_name}/safety_dual_mu"] = float(safety_dual_mu)
        channel_metrics[f"pdpo/channel/{metric_name}/safety_dual_scale"] = float(safety_dual_scale)
        channel_metrics[f"pdpo/channel/{metric_name}/safety_dual_violation"] = float(safety_dual_violation)
        channel_metrics[f"pdpo/channel/{metric_name}/safety_dual_pressure"] = float(safety_dual_pressure)
        channel_metrics[f"pdpo/channel/{metric_name}/safety_dual_pressure_ema"] = float(safety_dual_pressure_ema)
        channel_metrics[f"pdpo/channel/{metric_name}/safety_dual_updated"] = float(safety_dual_updated)
        channel_metrics[f"pdpo/channel/{metric_name}/batch_reliability"] = reliability_stats["batch_reliability"]
        channel_metrics[f"pdpo/channel/{metric_name}/correct_mean"] = reliability_stats["correct_mean"]
        channel_metrics[f"pdpo/channel/{metric_name}/wrong_mean"] = reliability_stats["wrong_mean"]
        channel_metrics[f"pdpo/channel/{metric_name}/correct_minus_wrong"] = reliability_stats["correct_minus_wrong"]
        channel_metrics[f"pdpo/channel/{metric_name}/wrong_high_rate"] = reliability_stats["wrong_high_rate"]
        channel_metrics[f"pdpo/channel/{metric_name}/supports_correct_rate"] = reliability_stats["supports_correct_rate"]
        channel_metrics[f"pdpo/channel/{metric_name}/pairwise_alignment_rate"] = reliability_stats["pairwise_alignment_rate"]
        channel_metrics[f"pdpo/channel/{metric_name}/pairwise_inversion_rate"] = reliability_stats["pairwise_inversion_rate"]
        channel_metrics[f"pdpo/channel/{metric_name}/pairwise_tie_rate"] = reliability_stats["pairwise_tie_rate"]
        channel_metrics[f"pdpo/channel/{metric_name}/pairwise_total"] = reliability_stats["pairwise_total"]
        channel_metrics[f"pdpo/channel/{metric_name}/pairwise_reliability"] = reliability_stats.get("pairwise_reliability", 1.0)
        channel_metrics[f"pdpo/channel/{metric_name}/inversion_reliability"] = reliability_stats.get("inversion_reliability", 1.0)
        channel_metrics[f"pdpo/channel/{metric_name}/comparable_groups"] = reliability_stats["comparable_groups"]
        channel_metrics[f"pdpo/channel/{metric_name}/mean"] = float(aux_scores.mean().item())

        if preference_weight <= 0.0:
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
        aux_adv_sum = aux_adv_sum + preference_weight * aux_adv
        aux_raw_sum = aux_raw_sum + preference_weight * aux_scores
        active_channels += 1
        weight_sum += preference_weight

        active_group_count += sum(
            1 for idxs in id2indices.values()
            if len(idxs) > 1 and aux_scores[torch.tensor(idxs, device=aux_scores.device)].std().item() > cfg.min_aux_std
        )

    lambda_aux_effective = _effective_lambda_aux(state, cfg)
    aux_component = lambda_aux_effective * _lexicographic_aux_component(
        aux_adv_sum,
        main_scores,
        id2indices,
        cfg,
        eps,
    )
    combined_adv, safety_clamp_count, correctness_margin_min = _combine_correctness_safe(
        main_adv,
        aux_component,
        main_scores,
        id2indices,
        cfg,
        eps,
    )

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
        "pdpo/lambda_aux_start": cfg.lambda_aux_start,
        "pdpo/lambda_aux_warmup_steps": float(cfg.lambda_aux_warmup_steps),
        "pdpo/lambda_aux_effective": float(lambda_aux_effective),
        "pdpo/correctness_safe": float(cfg.correctness_safe),
        "pdpo/correctness_safe_clamp_count": float(safety_clamp_count),
        "pdpo/correctness_margin_min": float(correctness_margin_min),
        "pdpo/reliability_enabled": float(cfg.reliability_enabled),
        "pdpo/safety_dual_enabled": float(cfg.safety_dual_enabled),
        "pdpo/answer_gate_as_constraint": float(cfg.answer_gate_as_constraint),
        "pdpo/answer_gate_preference_scale": float(cfg.answer_gate_preference_scale),
        "pdpo/step": float(state.step),
    }
    if gate_scores is not None:
        PDPO_METRICS["pdpo/answer_gate_open_ratio"] = float((gate_scores >= cfg.answer_gate_min).float().mean().item())
    PDPO_METRICS.update(channel_metrics)

    return advantages, advantages
