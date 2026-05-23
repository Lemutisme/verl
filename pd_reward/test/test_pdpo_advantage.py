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
    import pdpo_init  # noqa: F401
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    return get_adv_estimator_fn("pdpo")


def test_pdpo_is_registered_by_pdpo_init():
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


def test_mixed_group_without_main_ties_ignores_aux_preference_signal():
    compute_pdpo_advantage = _load_pdpo()

    main_only_adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0]),
        response_mask=torch.ones(2, 4),
        index=np.array(["g1", "g1"]),
        pdpo_aux_rewards_dict={},
        pdpo_config_dict={
            "pdpo_eta_s": "0.0",
        },
    )

    reset_module = sys.modules.get("pdpo_advantage")
    if reset_module is not None:
        reset_module.reset_pdpo_state()

    aux_adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0]),
        response_mask=torch.ones(2, 4),
        index=np.array(["g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [1.0, 0.0],
        },
        pdpo_config_dict={
            "pdpo_beta_tie": "1.0",
            "pdpo_lambda_aux": "10.0",
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "pdpo_safety_dual_enabled": "false",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    assert torch.allclose(aux_adv[:, -1], main_only_adv[:, -1], atol=1e-6)


def test_mixed_group_uses_aux_only_inside_same_main_reward_bucket():
    compute_pdpo_advantage = _load_pdpo()

    adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 0.0, 1.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict={
            "pdpo_beta_tie": "1.0",
            "pdpo_lambda_aux": "1.0",
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "pdpo_safety_dual_enabled": "false",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    scalars = adv[:, -1]
    assert scalars[1].item() > scalars[0].item()
    assert scalars[3].item() > scalars[2].item()
    assert min(scalars[2].item(), scalars[3].item()) > max(scalars[0].item(), scalars[1].item())


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


def test_answer_extractability_gate_suppresses_other_aux_channels():
    compute_pdpo_advantage = _load_pdpo()

    adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 0.0, 0.0, 0.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [1.0, 1.0, 0.0, 0.0],
            "math_answer_extractability_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict={
            "pdpo_beta_same": "1.0",
            "pdpo_beta_tie": "0.0",
            "pdpo_eta_s": "0.0",
            "math_weight_step_arithmetic_validity_reward": "1.0",
            "math_weight_answer_extractability_reward": "0.0",
        },
    )

    scalars = adv[:, -1]
    assert scalars[1].item() > scalars[0].item()
    assert scalars[0].item() <= 0.0


def test_strict_correctness_safety_blocks_strong_antialigned_aux():
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
            "pdpo_beta_tie": "1.0",
            "pdpo_lambda_aux": "10.0",
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    scalars = adv[:, -1]
    assert scalars[1].item() > scalars[0].item()
    assert scalars[1].item() > scalars[2].item()
    assert scalars[3].item() > scalars[0].item()
    assert scalars[3].item() > scalars[2].item()


def test_strict_correctness_safety_keeps_aux_order_within_same_bucket():
    compute_pdpo_advantage = _load_pdpo()

    adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 0.0, 1.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict={
            "pdpo_beta_same": "1.0",
            "pdpo_beta_tie": "1.0",
            "pdpo_lambda_aux": "10.0",
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    scalars = adv[:, -1]
    assert scalars[1].item() > scalars[0].item()
    assert scalars[3].item() > scalars[2].item()
    assert min(scalars[2].item(), scalars[3].item()) > max(scalars[0].item(), scalars[1].item())


def test_antialigned_channel_reliability_drops_effective_weight():
    compute_pdpo_advantage = _load_pdpo()

    compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [1.0, 0.0, 1.0, 0.0],
        },
        pdpo_config_dict={
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_ema_alpha": "1.0",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    import pdpo_advantage

    metrics = pdpo_advantage.PDPO_METRICS
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/pairwise_inversion_rate"] == pytest.approx(1.0)
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/pairwise_alignment_rate"] == pytest.approx(0.0)
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/reliability"] == pytest.approx(0.0)
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/effective_weight"] == pytest.approx(0.0)


def test_aligned_channel_reliability_keeps_effective_weight():
    compute_pdpo_advantage = _load_pdpo()

    compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict={
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_ema_alpha": "1.0",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    import pdpo_advantage

    metrics = pdpo_advantage.PDPO_METRICS
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/pairwise_inversion_rate"] == pytest.approx(0.0)
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/pairwise_alignment_rate"] == pytest.approx(1.0)
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/reliability"] == pytest.approx(1.0)
    assert metrics["pdpo/channel/math_step_arithmetic_validity_reward/effective_weight"] == pytest.approx(1.0)


def test_safety_dual_penalizes_antialigned_channel_without_reward_mixing():
    compute_pdpo_advantage = _load_pdpo()

    compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [1.0, 0.0, 1.0, 0.0],
        },
        pdpo_config_dict={
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "pdpo_safety_dual_enabled": "true",
            "pdpo_safety_dual_eta": "1.0",
            "pdpo_safety_dual_target_margin": "0.02",
            "pdpo_safety_dual_wrong_high_target": "0.20",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    import pdpo_advantage

    prefix = "pdpo/channel/math_step_arithmetic_validity_reward"
    metrics = pdpo_advantage.PDPO_METRICS
    assert metrics[f"{prefix}/safety_dual_violation"] > 0.0
    assert metrics[f"{prefix}/safety_dual_mu"] > 0.0
    assert 0.0 < metrics[f"{prefix}/safety_dual_scale"] < 1.0
    assert metrics[f"{prefix}/effective_weight"] < metrics[f"{prefix}/weight"]


def test_safety_dual_keeps_aligned_channel_unpenalized():
    compute_pdpo_advantage = _load_pdpo()

    compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict={
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "pdpo_safety_dual_enabled": "true",
            "pdpo_safety_dual_eta": "1.0",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    import pdpo_advantage

    prefix = "pdpo/channel/math_step_arithmetic_validity_reward"
    metrics = pdpo_advantage.PDPO_METRICS
    assert metrics[f"{prefix}/safety_dual_violation"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/safety_dual_mu"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/safety_dual_scale"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/effective_weight"] == pytest.approx(1.0)


def test_safety_dual_recovers_when_constraints_are_safely_satisfied():
    compute_pdpo_advantage = _load_pdpo()

    common_config = {
        "pdpo_eta_s": "0.0",
        "pdpo_reliability_enabled": "false",
        "pdpo_safety_dual_enabled": "true",
        "pdpo_safety_dual_eta": "1.0",
        "pdpo_safety_dual_recovery_scale": "1.0",
        "pdpo_safety_dual_ema_alpha": "1.0",
        "pdpo_safety_dual_min_comparable_groups": "1",
        "pdpo_safety_dual_target_margin": "0.02",
        "pdpo_safety_dual_wrong_high_target": "0.20",
        "math_weight_step_arithmetic_validity_reward": "1.0",
    }

    compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [1.0, 0.0, 1.0, 0.0],
        },
        pdpo_config_dict=common_config,
    )

    import pdpo_advantage

    prefix = "pdpo/channel/math_step_arithmetic_validity_reward"
    penalized_mu = pdpo_advantage.PDPO_METRICS[f"{prefix}/safety_dual_mu"]
    assert penalized_mu > 0.0

    compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict=common_config,
    )

    recovered_mu = pdpo_advantage.PDPO_METRICS[f"{prefix}/safety_dual_mu"]
    assert recovered_mu < penalized_mu


def test_safety_dual_does_not_update_below_min_comparable_groups():
    compute_pdpo_advantage = _load_pdpo()

    compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_step_arithmetic_validity_reward": [1.0, 0.0, 1.0, 0.0],
        },
        pdpo_config_dict={
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "pdpo_safety_dual_enabled": "true",
            "pdpo_safety_dual_eta": "1.0",
            "pdpo_safety_dual_min_comparable_groups": "2",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    import pdpo_advantage

    prefix = "pdpo/channel/math_step_arithmetic_validity_reward"
    metrics = pdpo_advantage.PDPO_METRICS
    assert metrics[f"{prefix}/safety_dual_violation"] > 0.0
    assert metrics[f"{prefix}/safety_dual_updated"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/safety_dual_mu"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/effective_weight"] == pytest.approx(1.0)


def test_answer_gate_channel_defaults_to_constraint_not_preference_signal():
    compute_pdpo_advantage = _load_pdpo()

    adv, _ = compute_pdpo_advantage(
        token_level_rewards=_make_token_rewards([0.0, 0.0, 0.0, 0.0]),
        response_mask=torch.ones(4, 4),
        index=np.array(["g1", "g1", "g1", "g1"]),
        pdpo_aux_rewards_dict={
            "math_answer_extractability_reward": [0.0, 1.0, 0.0, 1.0],
        },
        pdpo_config_dict={
            "pdpo_beta_same": "1.0",
            "pdpo_eta_s": "0.0",
            "pdpo_reliability_enabled": "false",
            "pdpo_safety_dual_enabled": "false",
            "math_weight_answer_extractability_reward": "1.0",
        },
    )

    import pdpo_advantage

    prefix = "pdpo/channel/math_answer_extractability_reward"
    assert torch.allclose(adv[:, -1], torch.zeros(4), atol=1e-6)
    assert pdpo_advantage.PDPO_METRICS[f"{prefix}/preference_weight"] == pytest.approx(0.0)


def test_lambda_aux_warmup_starts_from_configured_floor():
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
            "pdpo_eta_s": "0.0",
            "pdpo_lambda_aux": "1.0",
            "pdpo_lambda_aux_start": "0.0",
            "pdpo_lambda_aux_warmup_steps": "10",
            "pdpo_reliability_enabled": "false",
            "pdpo_safety_dual_enabled": "false",
            "math_weight_step_arithmetic_validity_reward": "1.0",
        },
    )

    import pdpo_advantage

    assert torch.allclose(adv[:, -1], torch.zeros(4), atol=1e-6)
    assert pdpo_advantage.PDPO_METRICS["pdpo/lambda_aux_effective"] == pytest.approx(0.0)
