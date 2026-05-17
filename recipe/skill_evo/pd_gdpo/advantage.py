"""PD-GDPO advantage estimator.

Registers the ``pd_gdpo`` advantage estimator with verl. Unlike scalar reward
shaping, primal-dual control is applied *after* per-component group
normalization:

1. group-normalize the primary reward within each prompt group;
2. for each auxiliary component, form correctness-gated residuals
   ``1[r0 > g] * (s_k - tau_k)`` and group-normalize them independently;
3. aggregate ``A0 + sum_k rho(lambda_k) * A_k``;
4. batch-whiten the aggregated advantage.

The dual variables are updated once per batch *after* pricing, using the
centralized :class:`PrimalDualController`.
"""

import logging
from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est

from .controller import get_controller

logger = logging.getLogger(__name__)

# Component reward scalars are passed through the batch under this key prefix.
COMPONENT_PREFIX = "pdcomp__"


def _safe_float(x) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _group_normalize(values: torch.Tensor, index, epsilon: float) -> torch.Tensor:
    """Group-normalize a [B] tensor within prompt groups defined by ``index``.

    Singleton groups get an advantage of 0 (no within-group signal), matching
    verl's GRPO convention.
    """
    out = torch.zeros_like(values)
    if index is None:
        members_all = list(range(values.shape[0]))
        if len(members_all) > 1:
            v = values
            out = (v - v.mean()) / (v.std() + epsilon)
        return out

    groups = defaultdict(list)
    for i, idx in enumerate(index):
        groups[idx].append(i)
    for members in groups.values():
        if len(members) <= 1:
            continue
        sel = torch.tensor(members, dtype=torch.long, device=values.device)
        v = values[sel]
        out[sel] = (v - v.mean()) / (v.std() + epsilon)
    return out


def _extract_components(data, device, dtype) -> dict:
    """Pull per-sample component scalars out of ``data.non_tensor_batch``."""
    components: dict[str, torch.Tensor] = {}
    if data is None or not hasattr(data, "non_tensor_batch"):
        return components
    for key, arr in data.non_tensor_batch.items():
        if not isinstance(key, str) or not key.startswith(COMPONENT_PREFIX):
            continue
        name = key[len(COMPONENT_PREFIX) :]
        vals = np.asarray([_safe_float(x) for x in arr], dtype=np.float64)
        components[name] = torch.tensor(vals, dtype=dtype, device=device)
    return components


@register_adv_est("pd_gdpo")
def compute_pd_gdpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    epsilon: float = 1e-6,
    config=None,
    data=None,
    **kwargs,
):
    """Compute PD-GDPO advantages.

    Args:
        token_level_rewards: ``(bs, response_length)``. The scalar primary
            reward sits at the last valid response token, so its row-sum is the
            per-sample primary reward ``r0``.
        response_mask: ``(bs, response_length)`` valid-token mask.
        index: ``(bs,)`` prompt-group ids (verl ``uid``).
        data: the full ``DataProto``; component scalars are read from
            ``data.non_tensor_batch`` keys prefixed with ``pdcomp__``.

    Returns:
        ``(advantages, returns)`` both ``(bs, response_length)``.
    """
    with torch.no_grad():
        primary = token_level_rewards.sum(dim=-1)  # [B]
        primary_adv = _group_normalize(primary, index, epsilon)

        components = _extract_components(data, primary.device, primary.dtype)

        if not components:
            logger.warning(
                "pd_gdpo: no '%s*' component tensors found in batch; "
                "falling back to plain GRPO on the primary reward.",
                COMPONENT_PREFIX,
            )
            adv = primary_adv.unsqueeze(-1) * response_mask
            return adv, adv

        controller = get_controller()
        component_names = sorted(components.keys())
        lambdas, taus = controller.get_state(component_names)
        rho = controller.rho(lambdas)
        gate = (primary > controller.config.correctness_gate).to(primary.dtype)

        raw_adv = primary_adv.clone()
        for name in component_names:
            residual = gate * (components[name] - taus[name])
            comp_adv = _group_normalize(residual, index, epsilon)
            raw_adv = raw_adv + rho[name] * comp_adv

        advantages = raw_adv.unsqueeze(-1) * response_mask
        advantages = verl_F.masked_whiten(advantages, response_mask) * response_mask

        # Dual update happens AFTER pricing the current batch with the old
        # lambda, so the update is independent of within-batch sample order.
        metrics = controller.update_duals(
            primary=primary.detach().cpu().tolist(),
            components={n: components[n].detach().cpu().tolist() for n in component_names},
            gate=gate.detach().cpu().tolist(),
        )
        if metrics:
            logger.info(
                "pd_gdpo step %d | %s",
                int(metrics.get("pd_gdpo/step", 0)),
                " ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in sorted(metrics.items())
                    if k.startswith("pd_gdpo/lambda_") or k == "pd_gdpo/ema_primary"
                ),
            )

        return advantages, advantages
