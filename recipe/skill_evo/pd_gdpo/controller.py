"""Centralized primal-dual controller for PD-GDPO.

The controller maintains, for a dynamically discovered set of auxiliary reward
components:

* dual variables ``lambda_k`` (Lagrange multipliers);
* adaptive targets ``tau_k`` (constraint levels);
* EMA statistics of the primary reward and of each component.

It is a process-global singleton. The advantage estimator (and hence the dual
update) runs on the single driver process inside ``compute_advantage``, so a
plain in-process singleton with a lock is sufficient -- no Ray actor is needed.

Configuration is read from environment variables (prefix ``PDGDPO_``), matching
the env-knob style already used by ``run_grpo.sh``.
"""

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _to_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _sigmoid(z: float) -> float:
    if z >= 40:
        return 1.0
    if z <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class ComponentConfig:
    """Per-component primal-dual hyperparameters."""

    tau_min: float = 0.20
    tau_max: float = 0.85
    eta: float = 0.05
    lambda_max: float = 4.0
    lambda_init: float = 0.0


@dataclass
class ControllerConfig:
    """Global primal-dual controller hyperparameters."""

    correctness_gate: float = 0.0
    perf_lo: float = 0.20
    perf_hi: float = 0.90
    ema_alpha: float = 0.05
    warmup_steps: int = 10
    warmup_alpha: float = 0.30
    eta_gate_center: float = 0.40
    eta_gate_scale: float = 8.0
    eta_decay: bool = True
    rho_mode: str = "raw"  # "raw": rho(lambda)=lambda; "dual_mass": lambda/(1+sum lambda)
    adaptive_tau: bool = True
    dual_update: str = "additive"  # "additive" or "mirror"
    default_component: ComponentConfig = field(default_factory=ComponentConfig)
    state_path: Optional[str] = None


def _env_component_config(name: str, default: ComponentConfig) -> ComponentConfig:
    """Resolve per-component config from PDGDPO_* env vars, falling back to defaults."""
    key = name.upper()
    return ComponentConfig(
        tau_min=_to_float(os.environ.get(f"PDGDPO_TAU_{key}_MIN"), default.tau_min),
        tau_max=_to_float(os.environ.get(f"PDGDPO_TAU_{key}_MAX"), default.tau_max),
        eta=_to_float(os.environ.get(f"PDGDPO_ETA_{key}"), default.eta),
        lambda_max=_to_float(os.environ.get(f"PDGDPO_LAMBDA_{key}_MAX"), default.lambda_max),
        lambda_init=_to_float(os.environ.get(f"PDGDPO_LAMBDA_{key}_INIT"), default.lambda_init),
    )


def load_controller_config_from_env() -> ControllerConfig:
    """Build a ControllerConfig from PDGDPO_* environment variables."""
    default_component = ComponentConfig(
        tau_min=_to_float(os.environ.get("PDGDPO_DEFAULT_TAU_MIN"), 0.20),
        tau_max=_to_float(os.environ.get("PDGDPO_DEFAULT_TAU_MAX"), 0.85),
        eta=_to_float(os.environ.get("PDGDPO_DEFAULT_ETA"), 0.05),
        lambda_max=_to_float(os.environ.get("PDGDPO_DEFAULT_LAMBDA_MAX"), 4.0),
        lambda_init=_to_float(os.environ.get("PDGDPO_DEFAULT_LAMBDA_INIT"), 0.0),
    )
    return ControllerConfig(
        correctness_gate=_to_float(os.environ.get("PDGDPO_CORRECTNESS_GATE"), 0.0),
        perf_lo=_to_float(os.environ.get("PDGDPO_PERF_LO"), 0.20),
        perf_hi=_to_float(os.environ.get("PDGDPO_PERF_HI"), 0.90),
        ema_alpha=_to_float(os.environ.get("PDGDPO_EMA_ALPHA"), 0.05),
        warmup_steps=_to_int(os.environ.get("PDGDPO_WARMUP_STEPS"), 10),
        warmup_alpha=_to_float(os.environ.get("PDGDPO_WARMUP_ALPHA"), 0.30),
        eta_gate_center=_to_float(os.environ.get("PDGDPO_ETA_GATE_CENTER"), 0.40),
        eta_gate_scale=_to_float(os.environ.get("PDGDPO_ETA_GATE_SCALE"), 8.0),
        eta_decay=_to_bool(os.environ.get("PDGDPO_ETA_DECAY"), True),
        rho_mode=str(os.environ.get("PDGDPO_RHO_MODE", "raw")).strip().lower(),
        adaptive_tau=_to_bool(os.environ.get("PDGDPO_ADAPTIVE_TAU"), True),
        dual_update=str(os.environ.get("PDGDPO_DUAL_UPDATE", "additive")).strip().lower(),
        default_component=default_component,
        state_path=os.environ.get("PDGDPO_STATE_PATH") or None,
    )


# Lambda is multiplicatively updated in mirror mode; a lambda stuck at exactly 0
# can never grow. Floor it to a tiny positive value before a mirror step.
_MIRROR_LAMBDA_FLOOR = 1e-3


class PrimalDualController:
    """Maintains dual state for PD-GDPO. Thread-safe; updated once per batch."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._lock = threading.Lock()
        self.lambdas: Dict[str, float] = {}
        self.taus: Dict[str, float] = {}
        self.ema_components: Dict[str, float] = {}
        self.ema_primary: float = 0.0
        self.step: int = 0
        self._cfgs: Dict[str, ComponentConfig] = {}

        if config.state_path and os.path.exists(config.state_path):
            try:
                with open(config.state_path) as f:
                    self.load_state_dict(json.load(f))
                logger.info("PD-GDPO controller resumed state from %s", config.state_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("PD-GDPO failed to load controller state: %s", e)

    def _register(self, name: str) -> None:
        if name in self._cfgs:
            return
        cfg = _env_component_config(name, self.config.default_component)
        if cfg.tau_max < cfg.tau_min:
            cfg.tau_max = cfg.tau_min
        self._cfgs[name] = cfg
        self.lambdas.setdefault(name, _clip(cfg.lambda_init, 0.0, cfg.lambda_max))
        self.ema_components.setdefault(name, 0.0)
        self.taus.setdefault(name, cfg.tau_min)

    def _adaptive_tau(self, name: str) -> float:
        cfg = self._cfgs[name]
        if not self.config.adaptive_tau:
            return cfg.tau_max
        lo, hi = self.config.perf_lo, self.config.perf_hi
        if hi <= lo:
            ratio = 1.0 if self.ema_primary > lo else 0.0
        else:
            ratio = _clip((self.ema_primary - lo) / (hi - lo), 0.0, 1.0)
        return cfg.tau_min + (cfg.tau_max - cfg.tau_min) * ratio

    def _eta(self, base_eta: float) -> float:
        if base_eta <= 0:
            return 0.0
        gate = _sigmoid(self.config.eta_gate_scale * (self.ema_primary - self.config.eta_gate_center))
        decay = 1.0 / math.sqrt(self.step + 1.0) if self.config.eta_decay else 1.0
        return base_eta * gate * decay

    def get_state(self, component_names: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (lambdas, taus) for the requested components, registering new ones."""
        with self._lock:
            for name in component_names:
                self._register(name)
            lambdas = {name: self.lambdas[name] for name in component_names}
            taus = {name: self._adaptive_tau(name) for name in component_names}
            return lambdas, taus

    def rho(self, lambdas: Dict[str, float]) -> Dict[str, float]:
        """Map dual variables to advantage-aggregation weights."""
        if self.config.rho_mode == "dual_mass":
            denom = 1.0 + sum(lambdas.values())
            return {k: v / denom for k, v in lambdas.items()}
        return dict(lambdas)

    def update_duals(
        self,
        primary: List[float],
        components: Dict[str, List[float]],
        gate: List[float],
    ) -> Dict[str, float]:
        """Update dual variables and EMA statistics once per rollout batch.

        Constraint estimates use only correctness-gated samples, matching the
        gated residuals used by the advantage estimator.
        """
        if not primary or not components:
            return {}

        with self._lock:
            n_gated = sum(1 for g in gate if g > 0)
            avg_primary = sum(primary) / float(len(primary))

            # Gated constraint estimate C_hat per component.
            chat: Dict[str, float] = {}
            for name, vals in components.items():
                if n_gated > 0:
                    chat[name] = sum(v for v, g in zip(vals, gate) if g > 0) / float(n_gated)
                else:
                    # No gated samples this batch: fall back to ungated mean.
                    chat[name] = sum(vals) / float(len(vals))

            alpha = self.config.warmup_alpha if self.step < self.config.warmup_steps else self.config.ema_alpha
            if self.step == 0:
                self.ema_primary = _clip(avg_primary, 0.0, 1.0)
                for name in components:
                    self.ema_components[name] = _clip(chat[name], 0.0, 1.0)
            else:
                self.ema_primary = _clip((1.0 - alpha) * self.ema_primary + alpha * avg_primary, 0.0, 1.0)
                for name in components:
                    prev = self.ema_components.get(name, 0.0)
                    self.ema_components[name] = _clip((1.0 - alpha) * prev + alpha * chat[name], 0.0, 1.0)

            self.step += 1

            metrics: Dict[str, float] = {}
            for name in components:
                self._register(name)
                cfg = self._cfgs[name]
                tau = self._adaptive_tau(name)
                self.taus[name] = tau
                eta = self._eta(cfg.eta)
                # violation > 0 means the constraint is under-satisfied.
                violation = tau - self.ema_components[name]
                lam = self.lambdas[name]
                if self.config.dual_update == "mirror":
                    lam = max(lam, _MIRROR_LAMBDA_FLOOR) * math.exp(eta * violation)
                else:
                    lam = lam + eta * violation
                lam = _clip(lam, 0.0, cfg.lambda_max)
                self.lambdas[name] = lam

                metrics[f"pd_gdpo/lambda_{name}"] = lam
                metrics[f"pd_gdpo/tau_{name}"] = tau
                metrics[f"pd_gdpo/chat_{name}"] = chat[name]
                metrics[f"pd_gdpo/ema_{name}"] = self.ema_components[name]
                metrics[f"pd_gdpo/violation_{name}"] = violation

            metrics["pd_gdpo/ema_primary"] = self.ema_primary
            metrics["pd_gdpo/gated_fraction"] = n_gated / float(len(primary))
            metrics["pd_gdpo/step"] = float(self.step)

            self._maybe_dump()
            return metrics

    def state_dict(self) -> dict:
        return {
            "lambdas": dict(self.lambdas),
            "taus": dict(self.taus),
            "ema_components": dict(self.ema_components),
            "ema_primary": self.ema_primary,
            "step": self.step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.lambdas = dict(state.get("lambdas", {}))
        self.taus = dict(state.get("taus", {}))
        self.ema_components = dict(state.get("ema_components", {}))
        self.ema_primary = float(state.get("ema_primary", 0.0))
        self.step = int(state.get("step", 0))

    def _maybe_dump(self) -> None:
        if not self.config.state_path:
            return
        try:
            tmp = f"{self.config.state_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(self.state_dict(), f, indent=2)
            os.replace(tmp, self.config.state_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("PD-GDPO failed to dump controller state: %s", e)


_CONTROLLER: Optional[PrimalDualController] = None
_CONTROLLER_LOCK = threading.Lock()


def get_controller() -> PrimalDualController:
    """Return the process-global primal-dual controller, creating it on first use."""
    global _CONTROLLER
    if _CONTROLLER is None:
        with _CONTROLLER_LOCK:
            if _CONTROLLER is None:
                _CONTROLLER = PrimalDualController(load_controller_config_from_env())
    return _CONTROLLER


def reset_controller() -> None:
    """Drop the global controller (used for tests)."""
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        _CONTROLLER = None
