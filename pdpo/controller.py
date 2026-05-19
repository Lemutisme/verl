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
"""Primal-dual controller for PD-GDPO.

The controller is a single-writer singleton living on the trainer driver
process. It owns dual variables ``lambda_k``, per-component targets
``tau_k``, and the EMA statistics needed to update them.

Lifecycle:
    * Driver constructs / fetches the controller via :func:`get_controller`.
    * The ``pd_gdpo`` advantage estimator calls :meth:`get_state` to read
      the current ``(lambda, tau)`` snapshot used for *pricing* the batch.
    * After pricing, the estimator calls :meth:`update_duals` exactly once
      per batch, which moves ``lambda_k`` and updates EMA / step.

The controller deliberately does **not** live inside the reward manager:
the reward manager should emit raw component scalars; only the estimator
sees ``lambda``. This keeps the primal–dual control localised at the
advantage level and avoids the "scalar PD" failure mode the proposal
diagnoses.

Configuration precedence (highest first):
    1. Env vars ``PDGDPO_*``
    2. Hydra config block ``algorithm.pd_gdpo.*``
    3. Hard-coded defaults below.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


@dataclass
class ComponentConfig:
    tau_min: float = 0.20
    tau_max: float = 0.85
    eta: float = 0.05
    # lambda_max default lowered from 4.0 -> 2.0 in line with upstream
    # pd_reward (commit 2b9e9fa2 lowered it to 0.5 because their scalarized
    # reward was getting clipped to [-1,1]). In PD-GDPO the batch whitening
    # neutralizes absolute scale, but a small lambda_max still keeps primary
    # advantage proportionally dominant -- which is why we also default
    # rho_mode to ``dual_mass`` (primary weight = 1 + sum λ_j).
    lambda_max: float = 2.0
    lambda_init: float = 0.0
    monotone_tau: bool = False  # if true, tau never decreases inside this component


@dataclass
class ControllerConfig:
    correctness_gate: float = 0.0
    perf_lo: float = 0.20
    perf_hi: float = 0.90
    ema_alpha: float = 0.05
    warmup_steps: int = 50
    warmup_alpha: float = 0.30
    eta_gate_center: float = 0.40
    eta_gate_scale: float = 8.0
    eta_decay: bool = True
    # Default to ``dual_mass`` (rho_k = λ_k / (1 + Σ λ_j)) following upstream
    # pd_reward, which switched normalize_by_dual_mass to True for PD mode.
    # The motivation is "prevent sub-rewards from overwhelming the main
    # reward": with dual_mass, primary advantage stays proportionally
    # weighted as (1 + Σ λ_j) even when many constraints are violated.
    # In PD-GDPO, after batch whitening this is approximately equivalent to
    # ``raw`` for *small* λ but keeps primary dominant once λ grows.
    rho_mode: str = "dual_mass"     # "raw" | "dual_mass"
    dual_update: str = "additive"   # "additive" | "mirror"
    mirror_floor: float = 1e-3
    adv_clip: float = 5.0           # clamp per-component group-adv before lambda
    state_path: Optional[str] = None
    component_defaults: ComponentConfig = field(default_factory=ComponentConfig)
    per_component: dict[str, ComponentConfig] = field(default_factory=dict)

    @classmethod
    def from_hydra(cls, cfg: Any) -> "ControllerConfig":
        out = cls()
        if cfg is None:
            return cls._apply_env(out)

        for field_name in (
            "correctness_gate",
            "perf_lo",
            "perf_hi",
            "ema_alpha",
            "warmup_alpha",
            "eta_gate_center",
            "eta_gate_scale",
            "mirror_floor",
            "adv_clip",
        ):
            setattr(out, field_name, float(_cfg_get(cfg, field_name, getattr(out, field_name))))
        out.warmup_steps = int(_cfg_get(cfg, "warmup_steps", out.warmup_steps))
        out.eta_decay = bool(_cfg_get(cfg, "eta_decay", out.eta_decay))
        out.rho_mode = str(_cfg_get(cfg, "rho_mode", out.rho_mode))
        out.dual_update = str(_cfg_get(cfg, "dual_update", out.dual_update))
        sp = _cfg_get(cfg, "state_path", None)
        out.state_path = None if sp in (None, "") else str(sp)

        defaults = _cfg_get(cfg, "component_defaults", None)
        if defaults is not None:
            out.component_defaults = ComponentConfig(
                tau_min=float(_cfg_get(defaults, "tau_min", out.component_defaults.tau_min)),
                tau_max=float(_cfg_get(defaults, "tau_max", out.component_defaults.tau_max)),
                eta=float(_cfg_get(defaults, "eta", out.component_defaults.eta)),
                lambda_max=float(_cfg_get(defaults, "lambda_max", out.component_defaults.lambda_max)),
                lambda_init=float(_cfg_get(defaults, "lambda_init", out.component_defaults.lambda_init)),
                monotone_tau=bool(_cfg_get(defaults, "monotone_tau", out.component_defaults.monotone_tau)),
            )

        comps = _cfg_get(cfg, "components", None)
        if comps is not None:
            try:
                items = comps.items()  # DictConfig / dict
            except AttributeError:
                items = []
            for name, sub in items:
                d = out.component_defaults
                out.per_component[str(name)] = ComponentConfig(
                    tau_min=float(_cfg_get(sub, "tau_min", d.tau_min)),
                    tau_max=float(_cfg_get(sub, "tau_max", d.tau_max)),
                    eta=float(_cfg_get(sub, "eta", d.eta)),
                    lambda_max=float(_cfg_get(sub, "lambda_max", d.lambda_max)),
                    lambda_init=float(_cfg_get(sub, "lambda_init", d.lambda_init)),
                    monotone_tau=bool(_cfg_get(sub, "monotone_tau", d.monotone_tau)),
                )
        return cls._apply_env(out)

    @staticmethod
    def _apply_env(out: "ControllerConfig") -> "ControllerConfig":
        out.correctness_gate = _env_float("PDGDPO_CORRECTNESS_GATE", out.correctness_gate)
        out.perf_lo = _env_float("PDGDPO_PERF_LO", out.perf_lo)
        out.perf_hi = _env_float("PDGDPO_PERF_HI", out.perf_hi)
        out.ema_alpha = _env_float("PDGDPO_EMA_ALPHA", out.ema_alpha)
        out.warmup_steps = int(_env_float("PDGDPO_WARMUP_STEPS", out.warmup_steps))
        out.warmup_alpha = _env_float("PDGDPO_WARMUP_ALPHA", out.warmup_alpha)
        out.eta_gate_center = _env_float("PDGDPO_ETA_GATE_CENTER", out.eta_gate_center)
        out.eta_gate_scale = _env_float("PDGDPO_ETA_GATE_SCALE", out.eta_gate_scale)
        out.adv_clip = _env_float("PDGDPO_ADV_CLIP", out.adv_clip)
        if "PDGDPO_ETA_DECAY" in os.environ:
            out.eta_decay = os.environ["PDGDPO_ETA_DECAY"].strip().lower() in {"1", "true", "yes", "on"}
        out.rho_mode = _env_str("PDGDPO_RHO_MODE", out.rho_mode).strip().lower()
        out.dual_update = _env_str("PDGDPO_DUAL_UPDATE", out.dual_update).strip().lower()
        out.state_path = _env_str("PDGDPO_STATE_PATH", out.state_path or "") or None

        d = out.component_defaults
        d.tau_min = _env_float("PDGDPO_DEFAULT_TAU_MIN", d.tau_min)
        d.tau_max = _env_float("PDGDPO_DEFAULT_TAU_MAX", d.tau_max)
        d.eta = _env_float("PDGDPO_DEFAULT_ETA", d.eta)
        d.lambda_max = _env_float("PDGDPO_DEFAULT_LAMBDA_MAX", d.lambda_max)
        d.lambda_init = _env_float("PDGDPO_DEFAULT_LAMBDA_INIT", d.lambda_init)
        if "PDGDPO_DEFAULT_MONOTONE_TAU" in os.environ:
            d.monotone_tau = os.environ["PDGDPO_DEFAULT_MONOTONE_TAU"].strip().lower() in {"1", "true", "yes", "on"}
        return out


class PrimalDualController:
    """Centralised dual-state container.

    Thread-safe via a coarse lock; state mutations are O(K) and happen
    once per batch, so contention is irrelevant in practice.
    """

    def __init__(self, cfg: Optional[ControllerConfig] = None):
        self.cfg = cfg or ControllerConfig()
        self._lock = threading.Lock()
        self.lambdas: dict[str, float] = {}
        self.taus: dict[str, float] = {}
        self.ema_components: dict[str, float] = {}
        self.ema_primary: float = 0.0
        self.ema_primary_max: float = 0.0
        self.step: int = 0
        self._registered: set[str] = set()
        if self.cfg.state_path and os.path.exists(self.cfg.state_path):
            try:
                with open(self.cfg.state_path) as fh:
                    self.load_state_dict(json.load(fh))
            except Exception:  # noqa: BLE001 -- never fail training on stale state
                pass

    # ----- registration ----------------------------------------------------
    def _component_cfg(self, name: str) -> ComponentConfig:
        return self.cfg.per_component.get(name, self.cfg.component_defaults)

    def _ensure_component(self, name: str) -> None:
        if name in self._registered:
            return
        ccfg = self._component_cfg(name)
        self.lambdas.setdefault(name, float(ccfg.lambda_init))
        self.taus.setdefault(name, float(ccfg.tau_min))
        self.ema_components.setdefault(name, 0.0)
        self._registered.add(name)

    # ----- public read API -------------------------------------------------
    def get_state(self, component_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
        """Return ``(lambdas, taus)`` snapshots for the requested components."""
        with self._lock:
            for name in component_names:
                self._ensure_component(name)
            lambdas = {n: float(self.lambdas[n]) for n in component_names}
            taus = {n: float(self.taus[n]) for n in component_names}
        return lambdas, taus

    def rho(self, lambdas: dict[str, float]) -> dict[str, float]:
        """Apply the configured ρ(λ) mapping (raw or normalised dual mass)."""
        if self.cfg.rho_mode == "dual_mass":
            mass = 1.0 + sum(max(0.0, v) for v in lambdas.values())
            return {k: max(0.0, v) / mass for k, v in lambdas.items()}
        return {k: max(0.0, v) for k, v in lambdas.items()}

    # ----- public write API -----------------------------------------------
    def update_duals(
        self,
        primary: float,
        gated_means: dict[str, float],
        gate_fraction: float,
    ) -> dict[str, float]:
        """Move λ, τ, EMA based on one batch.

        Args:
            primary: batch-mean primary reward (used for EMA + τ schedule).
            gated_means: per-component mean over gated samples (raw s_k,
                NOT the residual).  Pass an empty dict to skip update for
                components with no gated samples.
            gate_fraction: fraction of samples that passed the correctness
                gate; logged for diagnostics.

        Returns:
            Metrics dict (flat, prefixed with ``pd_gdpo/``) for the trainer.
        """
        metrics: dict[str, float] = {}
        with self._lock:
            # Auto-register any component that appears in gated_means but has
            # not been seen yet. This lets callers update_duals without first
            # calling get_state (useful in tests and in callers that emit
            # components opportunistically).
            for name in gated_means:
                self._ensure_component(name)
            self.step += 1
            alpha = self.cfg.warmup_alpha if self.step <= self.cfg.warmup_steps else self.cfg.ema_alpha
            alpha = float(max(0.0, min(1.0, alpha)))

            self.ema_primary = (1.0 - alpha) * self.ema_primary + alpha * float(primary)
            self.ema_primary = max(0.0, min(1.0, self.ema_primary))
            self.ema_primary_max = max(self.ema_primary_max, self.ema_primary)

            denom = max(self.cfg.perf_hi - self.cfg.perf_lo, 1e-6)
            ratio_now = (self.ema_primary - self.cfg.perf_lo) / denom
            ratio_now = max(0.0, min(1.0, ratio_now))

            for name in sorted(self._registered):
                ccfg = self._component_cfg(name)

                # τ schedule (optionally monotone in primary EMA max).
                ref = self.ema_primary_max if ccfg.monotone_tau else self.ema_primary
                ratio = max(0.0, min(1.0, (ref - self.cfg.perf_lo) / denom))
                tau_new = ccfg.tau_min + (ccfg.tau_max - ccfg.tau_min) * ratio
                if ccfg.monotone_tau:
                    tau_new = max(tau_new, self.taus[name])
                self.taus[name] = tau_new

                # Component EMA update only when the batch had gated samples
                # for that component; otherwise leave it untouched to avoid
                # spurious dual movement.
                if name in gated_means:
                    chat = float(gated_means[name])
                    self.ema_components[name] = (
                        (1.0 - alpha) * self.ema_components[name] + alpha * chat
                    )
                    self.ema_components[name] = max(0.0, min(1.0, self.ema_components[name]))

                # η scheduling: sigmoid-gated by primary EMA, then optionally
                # log-decayed.  Upstream pd_reward (commit 2b9e9fa2) switched
                # from 1/sqrt(step) to 1/(1 + 0.1·ln(step+1)) because the
                # 1/sqrt schedule was killing dual updates too quickly --
                # by step 300 the multiplier is 0.058 (sqrt) vs 0.637 (log).
                eta = ccfg.eta
                if eta > 0:
                    gate = 1.0 / (1.0 + math.exp(
                        -self.cfg.eta_gate_scale * (self.ema_primary - self.cfg.eta_gate_center)
                    ))
                    eta = eta * gate
                    if self.cfg.eta_decay:
                        # Exact upstream formula: 1/(1 + 0.1·ln(step+1))
                        # with step already incremented for this update, so
                        # the first call sees `ln(2)` not `ln(1)`.
                        eta = eta / (1.0 + 0.1 * math.log(float(self.step) + 1.0))

                violation = self.taus[name] - self.ema_components[name]

                if self.cfg.dual_update == "mirror":
                    base = max(self.lambdas[name], self.cfg.mirror_floor)
                    new_lambda = base * math.exp(eta * violation)
                else:  # additive default
                    new_lambda = self.lambdas[name] + eta * violation
                new_lambda = max(0.0, min(ccfg.lambda_max, new_lambda))
                self.lambdas[name] = float(new_lambda)

                metrics[f"pd_gdpo/lambda_{name}"] = float(self.lambdas[name])
                metrics[f"pd_gdpo/tau_{name}"] = float(self.taus[name])
                metrics[f"pd_gdpo/chat_{name}"] = float(self.ema_components[name])
                metrics[f"pd_gdpo/violation_{name}"] = float(violation)

            metrics["pd_gdpo/ema_primary"] = float(self.ema_primary)
            metrics["pd_gdpo/ratio"] = float(ratio_now)
            metrics["pd_gdpo/gate_fraction"] = float(gate_fraction)
            metrics["pd_gdpo/step"] = float(self.step)

            if self.cfg.state_path:
                try:
                    with open(self.cfg.state_path, "w") as fh:
                        json.dump(self.state_dict(), fh)
                except OSError:
                    pass
        return metrics

    # ----- checkpointing ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "lambdas": dict(self.lambdas),
            "taus": dict(self.taus),
            "ema_components": dict(self.ema_components),
            "ema_primary": float(self.ema_primary),
            "ema_primary_max": float(self.ema_primary_max),
            "step": int(self.step),
            "registered": sorted(self._registered),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lambdas = {str(k): float(v) for k, v in state.get("lambdas", {}).items()}
        self.taus = {str(k): float(v) for k, v in state.get("taus", {}).items()}
        self.ema_components = {str(k): float(v) for k, v in state.get("ema_components", {}).items()}
        self.ema_primary = float(state.get("ema_primary", 0.0))
        self.ema_primary_max = float(state.get("ema_primary_max", self.ema_primary))
        self.step = int(state.get("step", 0))
        self._registered = set(state.get("registered", list(self.lambdas.keys())))


# ---------------------------------------------------------------------------
# Singleton accessors -- the controller is process-local on the driver. The
# advantage estimator is the only writer; treat this as a single-writer
# global.

_CONTROLLER: Optional[PrimalDualController] = None
_CONTROLLER_LOCK = threading.Lock()


def get_controller(config: Any = None) -> PrimalDualController:
    """Return the process-local controller, constructing it on first use.

    ``config`` may be ``None`` (defaults + env) or a hydra
    ``DictConfig``-style object pointed at ``algorithm.pd_gdpo``.
    """
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        if _CONTROLLER is None:
            _CONTROLLER = PrimalDualController(ControllerConfig.from_hydra(config))
        return _CONTROLLER


def reset_controller() -> None:
    """Drop the singleton. Primarily useful for tests."""
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        _CONTROLLER = None
