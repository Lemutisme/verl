"""Unit tests for PDAR core: dual variable updates + selective damping."""

import sys
import os

# Ensure pd_reward is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import pytest

from reward_score.pdar_core import (
    PDARConfig,
    PDARState,
    selective_damp,
    update_constraint_dual,
    update_sharpness_dual,
    group_normalize_scores,
    reset_pdar_state,
)


# =====================================================================
# Selective Damping Tests
# =====================================================================


class TestSelectiveDamp:
    """Test the bounded-influence sharpness damping function."""

    def test_identity_when_lambda_zero(self):
        """When lambda_s = 0, selective_damp should return the input unchanged."""
        adv = torch.tensor([1.0, -0.5, 0.0, 2.0, -1.5])
        result = selective_damp(adv, lambda_s=0.0)
        assert torch.allclose(result, adv, atol=1e-6)

    def test_sign_preserved(self):
        """Damping should preserve the sign of each advantage relative to the mean."""
        adv = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = selective_damp(adv, lambda_s=1.0)
        mean = adv.mean()
        # All deviations from mean should keep their sign
        original_signs = torch.sign(adv - mean)
        damped_signs = torch.sign(result - result.mean())
        # Zero deviations can flip sign, ignore them
        mask = original_signs != 0
        assert torch.all(original_signs[mask] == damped_signs[mask])

    def test_extreme_advantages_compressed(self):
        """Extreme advantages should be compressed more than small ones."""
        adv = torch.tensor([-10.0, -0.1, 0.0, 0.1, 10.0])
        result = selective_damp(adv, lambda_s=1.0)
        mean = adv.mean()
        std = adv.std() + 1e-6

        # The extreme values (±10) should be compressed significantly
        # while the near-mean values (-0.1, 0, 0.1) should be roughly unchanged
        original_range = adv.max() - adv.min()
        damped_range = result.max() - result.min()
        assert damped_range < original_range * 0.5, (
            f"Expected significant compression: original range {original_range:.2f}, "
            f"damped range {damped_range:.2f}"
        )

    def test_small_advantages_nearly_unchanged(self):
        """Advantages close to the mean should be nearly unchanged."""
        adv = torch.tensor([0.9, 0.95, 1.0, 1.05, 1.1])
        result = selective_damp(adv, lambda_s=0.5)
        # With small deviations and moderate lambda_s, values should be close
        assert torch.allclose(result, adv, atol=0.15)

    def test_ranking_approximately_preserved(self):
        """Approximate ranking should be preserved after damping."""
        adv = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
        result = selective_damp(adv, lambda_s=0.5)
        # Check monotonicity
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], (
                f"Ranking broken at index {i}: {result[i].item():.4f} > {result[i+1].item():.4f}"
            )

    def test_large_lambda_strong_compression(self):
        """Very large lambda_s should compress all advantages toward the mean."""
        adv = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
        result = selective_damp(adv, lambda_s=100.0)
        mean = result.mean()
        # All values should be very close to the mean
        assert (result - mean).abs().max() < 0.5


# =====================================================================
# Dual Variable Update Tests
# =====================================================================


class TestConstraintDualUpdate:
    """Test the constraint dual variable update rule."""

    def test_violation_increases_lambda(self):
        """When aux metric is below target, lambda_c should increase."""
        state = PDARState(lambda_c=0.0)
        config = PDARConfig(eta_c=0.1, tau_c=0.8, sign_c=1.0, lambda_c_max=5.0)
        update_constraint_dual(state, config, aux_mean=0.3)
        assert state.lambda_c > 0.0, "Lambda should increase when aux < target"

    def test_satisfaction_decreases_lambda(self):
        """When aux metric exceeds target, lambda_c should decrease (toward 0)."""
        state = PDARState(lambda_c=0.5)
        config = PDARConfig(eta_c=0.1, tau_c=0.5, sign_c=1.0, lambda_c_max=5.0)
        update_constraint_dual(state, config, aux_mean=0.9)
        assert state.lambda_c < 0.5, "Lambda should decrease when aux > target"

    def test_lambda_non_negative(self):
        """Lambda_c should never go below 0."""
        state = PDARState(lambda_c=0.01)
        config = PDARConfig(eta_c=1.0, tau_c=0.1, sign_c=1.0, lambda_c_max=5.0)
        update_constraint_dual(state, config, aux_mean=10.0)
        assert state.lambda_c >= 0.0

    def test_lambda_capped(self):
        """Lambda_c should be capped at lambda_c_max."""
        state = PDARState(lambda_c=0.0)
        config = PDARConfig(eta_c=100.0, tau_c=1.0, sign_c=1.0, lambda_c_max=2.0)
        update_constraint_dual(state, config, aux_mean=0.0)
        assert state.lambda_c <= 2.0

    def test_negative_sign_inverts_direction(self):
        """With sign_c=-1 (violation: lower is better), exceeding target should increase lambda."""
        state = PDARState(lambda_c=0.0)
        config = PDARConfig(eta_c=0.1, tau_c=0.3, sign_c=-1.0, lambda_c_max=5.0)
        # aux_mean = 0.5 > target 0.3, with sign=-1 this is a violation
        update_constraint_dual(state, config, aux_mean=0.5)
        assert state.lambda_c > 0.0


class TestSharpnessDualUpdate:
    """Test the sharpness dual variable update rule."""

    def test_high_sharpness_increases_lambda(self):
        """When sharpness exceeds target, lambda_s should increase."""
        state = PDARState(lambda_s=0.0, step=0)
        config = PDARConfig(eta_s=0.1, tau_s=1.0, lambda_s_max=5.0)
        update_sharpness_dual(state, config, current_sharpness=2.0)
        assert state.lambda_s > 0.0

    def test_low_sharpness_decreases_lambda(self):
        """When sharpness is below target, lambda_s should decrease."""
        state = PDARState(lambda_s=0.5, step=5)
        config = PDARConfig(eta_s=0.1, tau_s=2.0, lambda_s_max=5.0, sharpness_ema_alpha=1.0)
        update_sharpness_dual(state, config, current_sharpness=0.5)
        assert state.lambda_s < 0.5

    def test_ema_smoothing(self):
        """EMA should smooth the sharpness signal across steps."""
        state = PDARState(lambda_s=0.0, step=0)
        config = PDARConfig(eta_s=0.0, tau_s=100.0, lambda_s_max=5.0, sharpness_ema_alpha=0.1)

        # First observation: EMA should be exactly the observation
        update_sharpness_dual(state, config, current_sharpness=10.0)
        assert abs(state.sharpness_ema - 10.0) < 1e-6

        # Second observation: EMA should be smoothed
        update_sharpness_dual(state, config, current_sharpness=20.0)
        expected_ema = 0.9 * 10.0 + 0.1 * 20.0  # 11.0
        assert abs(state.sharpness_ema - expected_ema) < 1e-6

    def test_lambda_s_capped(self):
        """Lambda_s should be capped at lambda_s_max."""
        state = PDARState(lambda_s=0.0, step=0)
        config = PDARConfig(eta_s=100.0, tau_s=0.0, lambda_s_max=3.0, sharpness_ema_alpha=1.0)
        update_sharpness_dual(state, config, current_sharpness=100.0)
        assert state.lambda_s <= 3.0


# =====================================================================
# Group Normalization Tests
# =====================================================================


class TestGroupNormalize:
    """Test GRPO-style group normalization."""

    def test_single_group_mean_centered(self):
        """Within a single group, scores should be mean-centered."""
        import numpy as np
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = np.array(["g1", "g1", "g1", "g1"])
        result = group_normalize_scores(scores, index, norm_by_std=True)
        assert abs(result.mean().item()) < 1e-5

    def test_two_groups_independent(self):
        """Groups should be normalised independently."""
        import numpy as np
        scores = torch.tensor([10.0, 20.0, 100.0, 200.0])
        index = np.array(["a", "a", "b", "b"])
        result = group_normalize_scores(scores, index, norm_by_std=True)
        # Group a: (10-15)/std, (20-15)/std  →  centered
        # Group b: (100-150)/std, (200-150)/std  →  centered
        assert abs(result[0].item() + result[1].item()) < 1e-5  # group a sums to 0
        assert abs(result[2].item() + result[3].item()) < 1e-5  # group b sums to 0

    def test_single_element_group(self):
        """A group with one element should preserve its score (mean=0, std=1 convention)."""
        import numpy as np
        scores = torch.tensor([5.0, 10.0, 20.0])
        index = np.array(["solo", "pair", "pair"])
        result = group_normalize_scores(scores, index, norm_by_std=True)
        # Single-element group: GRPO convention is mean=0, std=1 → score unchanged
        assert abs(result[0].item() - 5.0) < 1e-4
        # Pair group should be mean-centered
        assert abs(result[1].item() + result[2].item()) < 1e-5


# =====================================================================
# PDARConfig Tests
# =====================================================================


class TestPDARConfig:
    """Test configuration parsing."""

    def test_from_dict_with_prefix(self):
        """Config should parse pdar_-prefixed keys."""
        d = {
            "pdar_eta_c": "0.1",
            "pdar_lambda_s_max": "3.0",
            "pdar_tau_s": "2.0",
            "unrelated_key": "ignored",
        }
        cfg = PDARConfig.from_dict(d)
        assert cfg.eta_c == 0.1
        assert cfg.lambda_s_max == 3.0
        assert cfg.tau_s == 2.0
        # Defaults should be preserved
        assert cfg.eta_s == 0.01

    def test_from_empty_dict(self):
        """Empty dict should return defaults."""
        cfg = PDARConfig.from_dict({})
        assert cfg.eta_c == 0.05
        assert cfg.eta_s == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
