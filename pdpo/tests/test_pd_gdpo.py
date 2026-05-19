# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke tests for the PD-GDPO recipe.

Run from repo root:

    python -m pytest recipe/pdpo/tests/test_pd_gdpo.py -xvs
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from recipe.pdpo.controller import ControllerConfig, PrimalDualController, get_controller, reset_controller


def test_controller_lambda_grows_on_violation():
    reset_controller()
    cfg = ControllerConfig()
    cfg.warmup_steps = 0
    cfg.ema_alpha = 1.0  # one-step EMA so the test is deterministic
    cfg.eta_decay = False
    cfg.eta_gate_center = -1.0  # gate always ~1
    cfg.component_defaults.eta = 0.5
    cfg.component_defaults.tau_min = 0.5
    cfg.component_defaults.tau_max = 0.5
    cfg.component_defaults.lambda_init = 0.0
    cfg.component_defaults.lambda_max = 4.0
    ctrl = PrimalDualController(cfg)

    # primary high enough that tau interpolates to tau_max=0.5
    ctrl.update_duals(primary=1.0, gated_means={"thought": 0.0}, gate_fraction=1.0)
    # violation = tau - chat = 0.5 - 0 = 0.5; λ ≈ 0 + 0.5 * gate * 0.5 (gate≈1, no decay)
    assert ctrl.lambdas["thought"] > 0.0, ctrl.lambdas


def test_controller_lambda_shrinks_when_satisfied():
    reset_controller()
    cfg = ControllerConfig()
    cfg.warmup_steps = 0
    cfg.ema_alpha = 1.0
    cfg.eta_decay = False
    cfg.eta_gate_center = -1.0
    cfg.component_defaults.eta = 0.5
    cfg.component_defaults.tau_min = 0.5
    cfg.component_defaults.tau_max = 0.5
    cfg.component_defaults.lambda_init = 1.0
    cfg.component_defaults.lambda_max = 4.0
    ctrl = PrimalDualController(cfg)

    ctrl.update_duals(primary=1.0, gated_means={"thought": 0.9}, gate_fraction=1.0)
    # violation = 0.5 - 0.9 = -0.4 -> λ decreases.
    assert ctrl.lambdas["thought"] < 1.0


def test_controller_state_dict_roundtrip():
    reset_controller()
    ctrl = PrimalDualController(ControllerConfig())
    ctrl.update_duals(primary=0.5, gated_means={"a": 0.4, "b": 0.6}, gate_fraction=1.0)
    state = ctrl.state_dict()

    other = PrimalDualController(ControllerConfig())
    other.load_state_dict(state)
    assert other.lambdas == ctrl.lambdas
    assert other.taus == ctrl.taus
    assert other.step == ctrl.step


def test_controller_rho_modes():
    reset_controller()
    ctrl = PrimalDualController(ControllerConfig(rho_mode="dual_mass"))
    out = ctrl.rho({"a": 1.0, "b": 3.0})
    # raw lambdas (1, 3) -> dual_mass (1/5, 3/5)
    assert out["a"] == pytest.approx(0.2)
    assert out["b"] == pytest.approx(0.6)

    raw_ctrl = PrimalDualController(ControllerConfig(rho_mode="raw"))
    assert raw_ctrl.rho({"a": 1.0, "b": 3.0}) == {"a": 1.0, "b": 3.0}


def test_estimator_falls_back_to_grpo_without_components():
    # Importing pkg triggers @register_adv_est("pd_gdpo").
    reset_controller()
    from recipe.pdpo.advantage import compute_pd_gdpo_outcome_advantage

    bsz, seq = 6, 4
    rewards = torch.zeros(bsz, seq)
    # Place primary at the last token of each sequence.
    primaries = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    rewards[:, -1] = primaries
    mask = torch.ones(bsz, seq)
    index = np.array(["p0", "p0", "p0", "p1", "p1", "p1"])

    adv, ret = compute_pd_gdpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        config=None,
        non_tensor_batch={},  # no components configured
        batch=None,
    )
    assert adv.shape == (bsz, seq)
    assert torch.allclose(adv, ret)
    # Each group has both pass and fail samples, so per-group adv is non-trivial.
    assert adv.abs().sum() > 0.0


def test_estimator_with_components_and_dual_update():
    reset_controller()
    from recipe.pdpo.advantage import compute_pd_gdpo_outcome_advantage
    from recipe.pdpo.controller import get_controller

    bsz, seq = 8, 3
    primaries = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    rewards = torch.zeros(bsz, seq)
    rewards[:, -1] = primaries
    mask = torch.ones(bsz, seq)
    index = np.array(["p0", "p0", "p0", "p0", "p1", "p1", "p1", "p1"])

    components = {
        "thought": np.array([0.9, 0.1, 0.9, 0.1, 0.2, 0.8, 0.2, 0.8], dtype=np.float32),
        "action": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    }

    class _Cfg:
        def __init__(self):
            self._d = {
                "pd_gdpo": {
                    "component_keys": ["thought", "action"],
                    "correctness_gate": 0.5,
                    "warmup_steps": 0,
                    "ema_alpha": 1.0,
                    "eta_decay": False,
                    "eta_gate_center": -1.0,
                    "component_defaults": {
                        "eta": 0.5,
                        "tau_min": 0.5,
                        "tau_max": 0.5,
                        "lambda_init": 0.5,
                        "lambda_max": 4.0,
                    },
                }
            }

        def get(self, k, d=None):
            v = self._d.get(k, d)
            if isinstance(v, dict):
                return _DictCfg(v)
            return v

    class _DictCfg:
        def __init__(self, d):
            self._d = d

        def get(self, k, d=None):
            v = self._d.get(k, d)
            if isinstance(v, dict):
                return _DictCfg(v)
            return v

        def items(self):
            return self._d.items()

    adv, _ = compute_pd_gdpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        config=_Cfg(),
        non_tensor_batch=components,
        batch=None,
    )
    assert adv.shape == (bsz, seq)
    assert torch.isfinite(adv).all()
    # After one batch with gated samples, λ for "thought" should have moved.
    ctrl = get_controller(None)
    assert "thought" in ctrl.lambdas
    # gated mean for thought = mean over passing samples (primaries==1):
    # samples 0,1 in group p0 (0.9, 0.1) and 4,5 in p1 (0.2, 0.8) -> 0.5
    # violation = tau (0.5) - chat (0.5) = 0.0 -> no λ movement
    assert ctrl.lambdas["thought"] == pytest.approx(0.5, abs=1e-6)
    # For "action" all values are 0.5 -> chat=0.5 -> no movement either.
    assert ctrl.lambdas["action"] == pytest.approx(0.5, abs=1e-6)


def test_estimator_lambda_grows_when_component_undershoots():
    reset_controller()
    from recipe.pdpo.advantage import compute_pd_gdpo_outcome_advantage
    from recipe.pdpo.controller import get_controller

    bsz, seq = 4, 2
    primaries = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rewards = torch.zeros(bsz, seq)
    rewards[:, -1] = primaries
    mask = torch.ones(bsz, seq)
    index = np.array(["a", "a", "b", "b"])
    components = {
        "thought": np.array([0.1, 0.2, 0.0, 0.1], dtype=np.float32),
    }

    cfg = {
        "pd_gdpo": {
            "component_keys": ["thought"],
            "correctness_gate": 0.0,
            "warmup_steps": 0,
            "ema_alpha": 1.0,
            "eta_decay": False,
            "eta_gate_center": -10.0,
            "component_defaults": {
                "eta": 0.4,
                "tau_min": 0.7,
                "tau_max": 0.7,
                "lambda_init": 0.0,
                "lambda_max": 4.0,
            },
        }
    }

    class _Dict:
        def __init__(self, d):
            self._d = d
        def get(self, k, d=None):
            v = self._d.get(k, d)
            return _Dict(v) if isinstance(v, dict) else v
        def items(self):
            return self._d.items()

    compute_pd_gdpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        config=_Dict(cfg),
        non_tensor_batch=components,
        batch=None,
    )
    ctrl = get_controller(None)
    assert ctrl.lambdas["thought"] > 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-xvs"]))
