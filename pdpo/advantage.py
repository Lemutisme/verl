# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PD-GDPO outcome advantage estimator.

The pipeline, in order, mirrors §4 of the proposal:

    1. Group-normalize the primary reward (standard GRPO).
    2. For each auxiliary component k, build the gated residual
       ``c_i^k = 1[r_i^0 > g] * (s_i^k - tau_k)`` and group-normalize it
       *independently*.
    3. Aggregate: ``A_i^raw = A_i^0 + sum_k rho(lambda_k) * A_i^k``,
       where ``rho`` is either identity or normalised dual mass.
    4. Batch whiten ``A_i^raw`` against ``response_mask``.
    5. After pricing, push gated component means + primary into the
       PrimalDualController (single batched dual step).

Per-component group advantages are clipped to a configurable band before
the lambda-weighted sum, so that nearly-singleton gates do not blow up
the aggregate (this addresses the edge case the proposal does not
specify).

The estimator follows the same registration / signature contract as
verl's built-in GDPO estimator, so it slots into ``compute_advantage``
with a one-line passthrough.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est

from .controller import get_controller


def _resolve_component_keys(config: Any, non_tensor_batch: dict) -> list[str]:
    """Pick the ordered list of auxiliary-component keys to consume.

    Priority:
        1. ``algorithm.pd_gdpo.component_keys`` (explicit, ordered).
        2. ``algorithm.gdpo_reward_keys`` (so GDPO configs work too).
        3. Any non-tensor key starting with ``pdcomp__`` (compat with the
           prior claude branch's convention).
    """
    if config is not None:
        pd = config.get("pd_gdpo", None) if hasattr(config, "get") else None
        keys = None
        if pd is not None and hasattr(pd, "get"):
            keys = pd.get("component_keys", None)
        if keys is None and hasattr(config, "get"):
            keys = config.get("gdpo_reward_keys", None)
        if keys:
            return [str(k) for k in keys]

    # Fallback: discover pdcomp__* keys in non_tensor_batch.
    if non_tensor_batch is None:
        return []
    return sorted(k for k in non_tensor_batch.keys() if isinstance(k, str) and k.startswith("pdcomp__"))


def _component_scalar(non_tensor_batch: dict, key: str, batch_size: int, device) -> torch.Tensor:
    """Read a per-sample scalar reward component and return ``[B]`` tensor."""
    if key not in non_tensor_batch:
        raise KeyError(
            f"pd_gdpo: component key '{key}' missing from non_tensor_batch. "
            f"Available: {sorted(non_tensor_batch.keys())}"
        )
    raw = np.asarray(non_tensor_batch[key], dtype=np.float32).reshape(-1)
    if raw.shape[0] != batch_size:
        raise ValueError(
            f"pd_gdpo: component '{key}' has length {raw.shape[0]} but batch size is {batch_size}."
        )
    return torch.from_numpy(raw).to(device=device)


def _group_normalize(scores: torch.Tensor, index: np.ndarray, eps: float) -> torch.Tensor:
    """Per-prompt-group z-score, singleton groups -> 0. Matches GRPO semantics."""
    bsz = scores.shape[0]
    if index is None:
        # No grouping info -- treat the whole batch as one group.
        mean = scores.mean()
        std = scores.std(unbiased=False)
        return (scores - mean) / (std + eps)

    id2vals: dict[Any, list[torch.Tensor]] = defaultdict(list)
    for i in range(bsz):
        id2vals[index[i]].append(scores[i])

    id2mean: dict[Any, torch.Tensor] = {}
    id2std: dict[Any, torch.Tensor] = {}
    for idx, vals in id2vals.items():
        if len(vals) <= 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device)
            id2std[idx] = torch.tensor(1.0, device=scores.device)
        else:
            stacked = torch.stack(vals)
            id2mean[idx] = stacked.mean()
            id2std[idx] = stacked.std()

    out = scores.clone()
    for i in range(bsz):
        out[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
    return out


@register_adv_est("pd_gdpo")
def compute_pd_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[Any] = None,
    non_tensor_batch: Optional[dict] = None,
    batch: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Primal-dual GDPO advantage.

    Args:
        token_level_rewards: ``(bs, response_length)`` primary reward placed
            at the EOS token (verl convention).  Used as both the primary
            signal and the source of the correctness gate.
        response_mask: ``(bs, response_length)`` valid-token mask.
        index: ``(bs,)`` prompt-group ids (from ``data.non_tensor_batch['uid']``).
        epsilon: numerical stability.
        norm_adv_by_std_in_grpo: kept for signature parity; PD-GDPO
            always normalises by std (zero-mean unit-variance per group)
            for primary AND for each component.
        config: algorithm config block; reads ``config.pd_gdpo.*``.
        non_tensor_batch: source of per-sample component scalars.
        batch: ``DataProto.batch`` view; unused but kept for signature
            parity with GDPO.

    Returns:
        ``(advantages, returns)`` with shape ``(bs, response_length)``.
        Both tensors are identical (outcome-level estimator).

    If no component keys are configured / discoverable, the estimator
    silently degrades to plain GRPO on the primary reward.
    """
    del norm_adv_by_std_in_grpo, batch, kwargs  # signature parity only
    device = token_level_rewards.device
    bsz = token_level_rewards.shape[0]

    pd_cfg = None
    if config is not None and hasattr(config, "get"):
        pd_cfg = config.get("pd_gdpo", None)

    controller = get_controller(pd_cfg)
    cfg = controller.cfg

    # ---- primary -----------------------------------------------------------
    primary = token_level_rewards.sum(dim=-1)
    primary_adv = _group_normalize(primary, index, epsilon)

    # ---- components -------------------------------------------------------
    component_keys = _resolve_component_keys(config, non_tensor_batch or {})
    if non_tensor_batch is None or not component_keys:
        # No components configured -> fall back to plain GRPO.  We still
        # broadcast the primary advantage and whiten to keep the contract
        # identical to GDPO's no-component branch.
        scalars = primary_adv.unsqueeze(-1) * response_mask
        whitened = verl_F.masked_whiten(scalars, response_mask) * response_mask
        return whitened, whitened

    lambdas, taus = controller.get_state(component_keys)
    rho = controller.rho(lambdas)

    gate = (primary > cfg.correctness_gate).to(dtype=primary.dtype)

    raw_adv = primary_adv.clone()
    gated_means: dict[str, float] = {}
    n_gated = float(gate.sum().item())

    for key in component_keys:
        s = _component_scalar(non_tensor_batch, key, bsz, device)
        residual = gate * (s - taus[key])
        comp_adv = _group_normalize(residual, index, epsilon)
        if cfg.adv_clip > 0:
            comp_adv = torch.clamp(comp_adv, min=-cfg.adv_clip, max=cfg.adv_clip)
        raw_adv = raw_adv + float(rho[key]) * comp_adv

        # Compute gated-mean of the raw component s_k for the dual update.
        if n_gated > 0:
            gated_sum = (gate * s).sum().item()
            gated_means[key] = float(gated_sum / n_gated)

    scalars = raw_adv.unsqueeze(-1) * response_mask
    whitened = verl_F.masked_whiten(scalars, response_mask) * response_mask

    # ---- dual update (post-pricing, once per batch) -----------------------
    primary_batch_mean = float(primary.mean().item()) if bsz > 0 else 0.0
    gate_fraction = float(n_gated / max(bsz, 1))
    controller.update_duals(
        primary=primary_batch_mean,
        gated_means=gated_means,
        gate_fraction=gate_fraction,
    )

    return whitened, whitened
