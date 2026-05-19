"""Standalone smoke harness for PD-GDPO that does not require a full verl
environment (no ray / omegaconf / tensordict / transformers needed).

It stubs the minimal slice of ``verl`` used by ``recipe.pdpo.advantage``
and exercises:

    * controller dual update direction (grows on violation, shrinks when
      satisfied);
    * estimator pipeline on a fake batch (fallback path with no
      components, plus a normal path with two components and a dual
      step);
    * state_dict roundtrip.

Run from repo root:

    python recipe/pdpo/tests/run_smoke.py
"""

from __future__ import annotations

import os
import sys
import types


# ---------- Minimal verl stubs ---------------------------------------------
def _install_stubs() -> None:
    if "verl" in sys.modules:
        return

    verl_pkg = types.ModuleType("verl")
    verl_pkg.__path__ = []  # behave like a package
    sys.modules["verl"] = verl_pkg

    utils_pkg = types.ModuleType("verl.utils")
    utils_pkg.__path__ = []
    sys.modules["verl.utils"] = utils_pkg

    tf_mod = types.ModuleType("verl.utils.torch_functional")

    def masked_whiten(values, mask, eps: float = 1e-8):
        import torch

        mask = mask.to(values.dtype)
        total = mask.sum()
        if total <= 0:
            return values
        mean = (values * mask).sum() / total
        var = ((values - mean) ** 2 * mask).sum() / total
        std = (var + eps).sqrt()
        return (values - mean) / std * mask

    tf_mod.masked_whiten = masked_whiten
    sys.modules["verl.utils.torch_functional"] = tf_mod

    trainer_pkg = types.ModuleType("verl.trainer")
    trainer_pkg.__path__ = []
    ppo_pkg = types.ModuleType("verl.trainer.ppo")
    ppo_pkg.__path__ = []
    core_mod = types.ModuleType("verl.trainer.ppo.core_algos")
    core_mod.ADV_ESTIMATOR_REGISTRY = {}

    def register_adv_est(name):
        def decorator(fn):
            core_mod.ADV_ESTIMATOR_REGISTRY[name] = fn
            return fn

        return decorator

    core_mod.register_adv_est = register_adv_est
    sys.modules["verl.trainer"] = trainer_pkg
    sys.modules["verl.trainer.ppo"] = ppo_pkg
    sys.modules["verl.trainer.ppo.core_algos"] = core_mod


_install_stubs()


# ---------- Make `recipe.pdpo` importable ----------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

# Ensure `recipe` itself is importable as a namespace package, even though
# the submodule does not ship an ``__init__.py``.
if "recipe" not in sys.modules:
    recipe_pkg = types.ModuleType("recipe")
    recipe_pkg.__path__ = [os.path.join(REPO_ROOT, "recipe")]
    sys.modules["recipe"] = recipe_pkg


def main() -> int:
    import numpy as np
    import torch

    from recipe.pdpo.controller import (
        ControllerConfig,
        PrimalDualController,
        get_controller,
        reset_controller,
    )

    # --- controller direction tests ----------------------------------------
    reset_controller()
    cfg = ControllerConfig()
    cfg.warmup_steps = 0
    cfg.ema_alpha = 1.0
    cfg.eta_decay = False
    cfg.eta_gate_center = -1.0
    cfg.component_defaults.eta = 0.5
    cfg.component_defaults.tau_min = 0.5
    cfg.component_defaults.tau_max = 0.5
    cfg.component_defaults.lambda_init = 0.0
    cfg.component_defaults.lambda_max = 4.0
    ctrl = PrimalDualController(cfg)
    ctrl.update_duals(primary=1.0, gated_means={"k": 0.0}, gate_fraction=1.0)
    assert ctrl.lambdas["k"] > 0.0, f"λ should grow on violation, got {ctrl.lambdas}"
    print("[ok] controller λ grows on violation:", ctrl.lambdas)

    cfg.component_defaults.lambda_init = 1.0
    ctrl2 = PrimalDualController(cfg)
    ctrl2.update_duals(primary=1.0, gated_means={"k": 0.9}, gate_fraction=1.0)
    assert ctrl2.lambdas["k"] < 1.0, f"λ should shrink on satisfaction, got {ctrl2.lambdas}"
    print("[ok] controller λ shrinks on satisfaction:", ctrl2.lambdas)

    # --- state roundtrip ----------------------------------------------------
    state = ctrl2.state_dict()
    ctrl3 = PrimalDualController(ControllerConfig())
    ctrl3.load_state_dict(state)
    assert ctrl3.lambdas == ctrl2.lambdas
    assert ctrl3.step == ctrl2.step
    print("[ok] state_dict roundtrip")

    # --- estimator fallback (no components) ---------------------------------
    reset_controller()
    from recipe.pdpo.advantage import compute_pd_gdpo_outcome_advantage

    bsz, seq = 6, 4
    primaries = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    rewards = torch.zeros(bsz, seq)
    rewards[:, -1] = primaries
    mask = torch.ones(bsz, seq)
    index = np.array(["p0"] * 3 + ["p1"] * 3)

    adv, ret = compute_pd_gdpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        config=None,
        non_tensor_batch={},
        batch=None,
    )
    assert adv.shape == (bsz, seq)
    assert torch.allclose(adv, ret)
    assert adv.abs().sum() > 0.0
    print("[ok] estimator fallback path produces non-zero advantages")

    # --- estimator with components + dual update ----------------------------
    reset_controller()

    class _Dict:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            v = self._d.get(k, default)
            return _Dict(v) if isinstance(v, dict) else v

        def items(self):
            return self._d.items()

    pd_cfg = {
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

    bsz, seq = 4, 2
    primaries = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rewards = torch.zeros(bsz, seq)
    rewards[:, -1] = primaries
    mask = torch.ones(bsz, seq)
    index = np.array(["a", "a", "b", "b"])
    components = {"thought": np.array([0.1, 0.2, 0.0, 0.1], dtype=np.float32)}

    adv, _ = compute_pd_gdpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        config=_Dict(pd_cfg),
        non_tensor_batch=components,
        batch=None,
    )
    assert torch.isfinite(adv).all()
    assert adv.shape == (bsz, seq)
    ctrl = get_controller(None)
    assert ctrl.lambdas["thought"] > 0.0, ctrl.lambdas
    print("[ok] estimator + dual update grew λ on under-satisfied component:", ctrl.lambdas)

    # --- log-decay schedule (synced from upstream 2b9e9fa2) ----------------
    reset_controller()
    cfg = ControllerConfig()
    cfg.warmup_steps = 0
    cfg.ema_alpha = 0.0  # freeze EMA so successive steps see the same violation
    cfg.eta_gate_center = -1.0  # gate ~1
    cfg.eta_decay = True
    cfg.component_defaults.eta = 0.1  # small base so we don't saturate
    cfg.component_defaults.tau_min = 1.0
    cfg.component_defaults.tau_max = 1.0  # constant violation = 1.0
    cfg.component_defaults.lambda_init = 0.0
    cfg.component_defaults.lambda_max = 1000.0  # no saturation in 20 steps
    ctrl = PrimalDualController(cfg)
    deltas = []
    prev = 0.0
    for _ in range(20):
        ctrl.update_duals(primary=0.5, gated_means={"x": 0.0}, gate_fraction=1.0)
        deltas.append(ctrl.lambdas["x"] - prev)
        prev = ctrl.lambdas["x"]
    # Each step's delta should be positive (constant violation) and
    # monotonically decreasing (log-decay).  Old 1/sqrt schedule had a
    # much steeper drop; log decay should be gentle.
    assert all(d > 0 for d in deltas), deltas
    assert deltas[-1] / deltas[0] > 0.5, (
        f"log decay should be gentle; ratio={deltas[-1]/deltas[0]:.3f}, expected >0.5"
    )
    print(f"[ok] log-eta decay is gentle: first={deltas[0]:.4f}, last={deltas[-1]:.4f}, ratio={deltas[-1]/deltas[0]:.3f}")

    # --- rho_mode=dual_mass default ----------------------------------------
    reset_controller()
    ctrl = PrimalDualController(ControllerConfig())
    assert ctrl.cfg.rho_mode == "dual_mass", ctrl.cfg.rho_mode
    print("[ok] default rho_mode is dual_mass")

    # --- registration check ------------------------------------------------
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    assert "pd_gdpo" in ADV_ESTIMATOR_REGISTRY
    print("[ok] pd_gdpo registered in ADV_ESTIMATOR_REGISTRY")

    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
