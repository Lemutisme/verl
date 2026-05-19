"""Integration tests for the PDAR advantage estimator."""

import sys
import os

# Ensure pd_reward is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest

from reward_score.pdar_core import reset_pdar_state, get_pdar_state


@pytest.fixture(autouse=True)
def clean_pdar_state():
    """Reset PDAR singleton state before each test."""
    reset_pdar_state()
    yield
    reset_pdar_state()


def _make_token_rewards(scores, response_length=8):
    """Create token_level_rewards tensor from per-response scalar scores.

    Places each score at the last position of each response.
    """
    bs = len(scores)
    tensor = torch.zeros(bs, response_length, dtype=torch.float32)
    for i, s in enumerate(scores):
        tensor[i, -1] = s
    return tensor


def _make_response_mask(bs, response_length=8):
    """Create an all-ones response mask."""
    return torch.ones(bs, response_length, dtype=torch.float32)


class TestPDARAdvantageEstimator:
    """Test the full compute_pdar_advantage function."""

    def test_registration(self):
        """PDAR should be registered with verl's core_algos registry."""
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn
        # Import pdar_init to trigger registration
        import pdar_init  # noqa: F401

        fn = get_adv_estimator_fn("pdar")
        assert fn is not None
        assert callable(fn)

    def test_no_aux_equals_grpo(self):
        """Without aux rewards, PDAR should produce GRPO-equivalent advantages."""
        import pdar_init  # noqa: F401
        from pdar_advantage import compute_pdar_advantage

        # 2 groups × 2 responses each
        main_scores = [0.0, 1.0, 0.0, 1.0]
        token_rewards = _make_token_rewards(main_scores)
        mask = _make_response_mask(4)
        index = np.array(["g1", "g1", "g2", "g2"])

        adv, ret = compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
            aux_rewards_tensor=None,  # no aux
        )

        assert adv.shape == (4, 8)
        # Within each group, the better response should have positive advantage
        assert adv[1, -1].item() > adv[0, -1].item()  # g1: score 1.0 > 0.0
        assert adv[3, -1].item() > adv[2, -1].item()  # g2: score 1.0 > 0.0

    def test_with_aux_modifies_advantage(self):
        """With aux rewards and lambda_c > 0, advantages should shift."""
        import pdar_init  # noqa: F401
        from pdar_advantage import compute_pdar_advantage

        # Force lambda_c to be positive by using get_pdar_state
        state, cfg = get_pdar_state({"pdar_eta_c": "0.0"})
        state.lambda_c = 0.5  # set directly

        main_scores = [0.0, 1.0, 0.0, 1.0]
        # Aux: sample 0 has high aux (0.9), sample 1 has low aux (0.1)
        aux_scores = [0.9, 0.1, 0.9, 0.1]

        token_rewards = _make_token_rewards(main_scores)
        aux_tensor = _make_token_rewards(aux_scores)
        mask = _make_response_mask(4)
        index = np.array(["g1", "g1", "g2", "g2"])

        adv_with_aux, _ = compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
            aux_rewards_tensor=aux_tensor,
        )

        # Now compute without aux
        reset_pdar_state()
        adv_without, _ = compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
            aux_rewards_tensor=None,
        )

        # With aux, sample 0 (high aux) should have higher advantage than without
        # The aux signal should have modified the advantage ordering
        diff = (adv_with_aux - adv_without).sum(dim=-1)
        assert not torch.allclose(diff, torch.zeros_like(diff), atol=1e-4), \
            "Aux signal should modify advantages when lambda_c > 0"

    def test_sharpness_damping_reduces_spread(self):
        """With lambda_s > 0, advantage spread should be reduced."""
        import pdar_init  # noqa: F401
        from pdar_advantage import compute_pdar_advantage

        state, cfg = get_pdar_state({"pdar_eta_s": "0.0"})
        state.lambda_s = 2.0  # strong damping

        # Create a group with very different scores
        main_scores = [0.0, 0.0, 0.0, 10.0]
        token_rewards = _make_token_rewards(main_scores)
        mask = _make_response_mask(4)
        index = np.array(["g1", "g1", "g1", "g1"])

        adv_damped, _ = compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
        )

        # Compute without damping
        reset_pdar_state()
        state, cfg = get_pdar_state({"pdar_eta_s": "0.0"})
        state.lambda_s = 0.0  # no damping

        adv_undamped, _ = compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
        )

        # The damped version should have smaller spread
        damped_range = adv_damped[:, -1].max() - adv_damped[:, -1].min()
        undamped_range = adv_undamped[:, -1].max() - adv_undamped[:, -1].min()
        assert damped_range < undamped_range, (
            f"Damped range ({damped_range:.4f}) should be < undamped ({undamped_range:.4f})"
        )

    def test_dual_variables_update(self):
        """Dual variables should update after each call."""
        import pdar_init  # noqa: F401
        from pdar_advantage import compute_pdar_advantage

        state, cfg = get_pdar_state({
            "pdar_eta_c": "0.1",
            "pdar_eta_s": "0.1",
            "pdar_tau_c": "0.8",
            "pdar_tau_s": "0.5",
        })

        # Very different scores → high sharpness
        main_scores = [0.0, 0.0, 1.0, 1.0]
        aux_scores = [0.1, 0.1, 0.1, 0.1]  # low aux → below tau_c
        token_rewards = _make_token_rewards(main_scores)
        aux_tensor = _make_token_rewards(aux_scores)
        mask = _make_response_mask(4)
        index = np.array(["g1", "g1", "g1", "g1"])

        # Initial state
        assert state.lambda_c == 0.0
        assert state.lambda_s == 0.0
        assert state.step == 0

        compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
            aux_rewards_tensor=aux_tensor,
        )

        # After one call, lambda_c should have increased (aux below target)
        assert state.lambda_c > 0.0, "Lambda_c should increase when aux < target"
        assert state.step == 1

    def test_metrics_populated(self):
        """PDAR_METRICS should be populated after each call."""
        import pdar_init  # noqa: F401
        from pdar_advantage import compute_pdar_advantage, PDAR_METRICS

        main_scores = [0.0, 1.0]
        token_rewards = _make_token_rewards(main_scores)
        mask = _make_response_mask(2)
        index = np.array(["g1", "g1"])

        compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
        )

        assert "pdar/lambda_c" in PDAR_METRICS
        assert "pdar/lambda_s" in PDAR_METRICS
        assert "pdar/main_reward_mean" in PDAR_METRICS

    def test_output_shape(self):
        """Output tensors should have the correct shape."""
        import pdar_init  # noqa: F401
        from pdar_advantage import compute_pdar_advantage

        bs, resp_len = 6, 16
        token_rewards = torch.zeros(bs, resp_len)
        token_rewards[:, -1] = torch.randn(bs)
        mask = torch.ones(bs, resp_len)
        index = np.array(["a", "a", "a", "b", "b", "b"])

        adv, ret = compute_pdar_advantage(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
        )

        assert adv.shape == (bs, resp_len)
        assert ret.shape == (bs, resp_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
