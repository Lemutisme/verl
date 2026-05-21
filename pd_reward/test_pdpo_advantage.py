"""Tests for the PDPO advantage estimator."""

import os
import sys

import numpy as np
import pytest
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def clean_pdpo_state():
    try:
        from pdpo_advantage import reset_pdpo_state
    except ImportError:
        yield
        return

    reset_pdpo_state()
    yield
    reset_pdpo_state()


def _make_token_rewards(scores, response_length=4):
    tensor = torch.zeros(len(scores), response_length, dtype=torch.float32)
    for i, score in enumerate(scores):
        tensor[i, -1] = float(score)
    return tensor


def _load_pdpo():
    import pdar_init  # noqa: F401
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    return get_adv_estimator_fn("pdpo")


def test_pdpo_is_registered_by_pdar_init():
    assert callable(_load_pdpo())


def test_all_same_main_reward_uses_aux_channels_for_advantage():
    compute_pdpo_advantage = _load_pdpo()

    adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 0.0, 0.0, 0.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict={
            "pdpo_beta_same": "1.0",
            "pdpo_beta_tie": "0.0",
            "pdpo_eta_s": "0.0",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    scalars = adv[:, -1]
    assert scalars[1].item() > scalars[0].item()
    assert scalars[3].item() > scalars[2].item()
    assert scalars.abs().sum().item() > 0.0


def test_main_reward_variance_keeps_correct_samples_above_antialigned_aux():
    compute_pdpo_advantage = _load_pdpo()

    adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [1.0, 0.0, 1.0, 0.0],
        },
        pdpo_config_dict={
            "pdpo_beta_same": "1.0",
            "pdpo_beta_tie": "0.1",
            "pdpo_eta_s": "0.0",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    scalars = adv[:, -1]
    assert scalars[1].item() > scalars[0].item()
    assert scalars[3].item() > scalars[2].item()


def test_zero_variance_aux_channel_is_ignored():
    compute_pdpo_advantage = _load_pdpo()

    adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 0.0, 0.0, 0.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [0.5, 0.5, 0.5, 0.5],
        },
        pdpo_config_dict={
            "pdpo_beta_same": "1.0",
            "pdpo_eta_s": "0.0",
            "pdpo_min_aux_std": "1e-6",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    assert torch.allclose(adv[:, -1], torch.zeros(4), atol=1e-6)
