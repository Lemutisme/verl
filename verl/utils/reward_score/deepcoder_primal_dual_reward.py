from dataclasses import asdict, dataclass
import math
from threading import Lock
from typing import Any, Dict, List, Tuple, Union

from .deepcoder_action_thought_reward import (
    _extract_code,
    _get_tests_deepcoder,
    _run_deepcoder_eval,
    compute_action_score_from_sums,
    compute_thought_score,
)
from .deepcoder_codeql_robust_reward import compute_codeql_robustness, normalize_codeql_scalar

@dataclass
class PrimalDualState:
    lambda_thought: float = 0.0
    lambda_action: float = 0.0
    ema_perf: float = 0.0
    ema_thought: float = 0.0
    ema_action: float = 0.0
    step: int = 0

@dataclass
class PrimalDualConfig:
    perf_gate: float = 0.0
    enable_thought: bool = True
    enable_action: bool = False # set False for DeepCoder by default since tracing is hard on isolated sandbox
    sandbox_url: str = "http://localhost:8080/sandbox"

    timeout_s: int = 10
    trace_timeout_s: int = 15

    M_top: int = 25
    w1: float = 0.7
    w2: float = 0.3

    u1: float = 0.5
    u2: float = 0.5
    kappa: float = 8.0

    tau_th_min: float = 0.20
    tau_th_max: float = 0.85
    tau_ac_min: float = 0.20
    tau_ac_max: float = 0.85
    perf_lo: float = 0.20
    perf_hi: float = 0.90

    eta_th0: float = 0.05
    eta_ac0: float = 0.05
    eta_gate_center: float = 0.40
    eta_gate_scale: float = 8.0
    lambda_th_max: float = 4.0
    lambda_ac_max: float = 4.0
    ema_alpha: float = 0.05

    update_dual: bool = True
    dual_update_on_gated: bool = False
    normalize_by_dual_mass: bool = False
    reset_dual_state: bool = False

    enable_codeql_subreward: bool = True
    codeql_subreward_weight: float = 0.10
    codeql_subreward_threshold: float = 0.82
    codeql_subreward_scale: float = 0.15
    codeql_require_ok: bool = True
    codeql_bin: str = ""
    codeql_timeout_s: int = 120
    codeql_workdir: str = ""


_PD_STATE = PrimalDualState()
_PD_LOCK = Lock()

def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)

def _to_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str): return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return default

def _to_float(v: Any, default: float) -> float:
    try: return float(v)
    except: return default

def _to_int(v: Any, default: int) -> int:
    try: return int(v)
    except: return default

def _sigmoid(z: float) -> float:
    if z >= 40: return 1.0
    if z <= -40: return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def _adaptive_tau(perf_ref: float, tau_min: float, tau_max: float, perf_lo: float, perf_hi: float) -> float:
    if perf_hi <= perf_lo:
        ratio = 1.0 if perf_ref > perf_lo else 0.0
    else:
        ratio = _clip((perf_ref - perf_lo) / (perf_hi - perf_lo), 0.0, 1.0)
    return tau_min + (tau_max - tau_min) * ratio

def _adaptive_eta(base_eta: float, perf_ref: float, gate_center: float, gate_scale: float, step: int) -> float:
    if base_eta <= 0: return 0.0
    gate = _sigmoid(gate_scale * (perf_ref - gate_center))
    decay = 1.0 / math.sqrt(float(step) + 1.0)
    return base_eta * gate * decay

def get_primal_dual_state() -> Dict[str, float]:
    with _PD_LOCK:
        return asdict(_PD_STATE)

def reset_primal_dual_state(lambda_thought: float = 0.0, lambda_action: float = 0.0) -> None:
    with _PD_LOCK:
        _PD_STATE.lambda_thought = max(0.0, float(lambda_thought))
        _PD_STATE.lambda_action = max(0.0, float(lambda_action))
        _PD_STATE.ema_perf = 0.0
        _PD_STATE.ema_thought = 0.0
        _PD_STATE.ema_action = 0.0
        _PD_STATE.step = 0

def _build_cfg(kwargs: Dict[str, Any]) -> PrimalDualConfig:
    cfg = PrimalDualConfig(
        perf_gate=_to_float(kwargs.get("perf_gate", 0.0), 0.0),
        enable_thought=_to_bool(kwargs.get("enable_thought", True), True),
        enable_action=_to_bool(kwargs.get("enable_action", False), False),
        sandbox_url=kwargs.get("sandbox_url") or "http://localhost:8080/sandbox",
        timeout_s=_to_int(kwargs.get("timeout_s", 10), 10),
        trace_timeout_s=_to_int(kwargs.get("trace_timeout_s", 15), 15),
        M_top=_to_int(kwargs.get("M_top", 25), 25),
        w1=_to_float(kwargs.get("w1", 0.7), 0.7),
        w2=_to_float(kwargs.get("w2", 0.3), 0.3),
        u1=_to_float(kwargs.get("u1", 0.5), 0.5),
        u2=_to_float(kwargs.get("u2", 0.5), 0.5),
        kappa=_to_float(kwargs.get("kappa", 8.0), 8.0),
        tau_th_min=_to_float(kwargs.get("tau_th_min", kwargs.get("tau_thought_min", 0.20)), 0.20),
        tau_th_max=_to_float(kwargs.get("tau_th_max", kwargs.get("tau_thought_max", 0.85)), 0.85),
        tau_ac_min=_to_float(kwargs.get("tau_ac_min", kwargs.get("tau_action_min", 0.20)), 0.20),
        tau_ac_max=_to_float(kwargs.get("tau_ac_max", kwargs.get("tau_action_max", 0.85)), 0.85),
        perf_lo=_to_float(kwargs.get("perf_lo", 0.20), 0.20),
        perf_hi=_to_float(kwargs.get("perf_hi", 0.90), 0.90),
        eta_th0=_to_float(kwargs.get("eta_th0", kwargs.get("eta_thought", 0.05)), 0.05),
        eta_ac0=_to_float(kwargs.get("eta_ac0", kwargs.get("eta_action", 0.05)), 0.05),
        eta_gate_center=_to_float(kwargs.get("eta_gate_center", 0.40), 0.40),
        eta_gate_scale=_to_float(kwargs.get("eta_gate_scale", 8.0), 8.0),
        lambda_th_max=_to_float(kwargs.get("lambda_th_max", kwargs.get("lambda_thought_max", 4.0)), 4.0),
        lambda_ac_max=_to_float(kwargs.get("lambda_ac_max", kwargs.get("lambda_action_max", 4.0)), 4.0),
        ema_alpha=_to_float(kwargs.get("ema_alpha", 0.05), 0.05),
        update_dual=_to_bool(kwargs.get("update_dual", True), True),
        dual_update_on_gated=_to_bool(kwargs.get("dual_update_on_gated", False), False),
        normalize_by_dual_mass=_to_bool(kwargs.get("normalize_by_dual_mass", False), False),
        reset_dual_state=_to_bool(kwargs.get("reset_dual_state", False), False),
        enable_codeql_subreward=_to_bool(kwargs.get("enable_codeql_subreward", True), True),
        codeql_subreward_weight=_to_float(kwargs.get("codeql_subreward_weight", 0.10), 0.10),
        codeql_subreward_threshold=_to_float(kwargs.get("codeql_subreward_threshold", 0.82), 0.82),
        codeql_subreward_scale=_to_float(kwargs.get("codeql_subreward_scale", 0.15), 0.15),
        codeql_require_ok=_to_bool(kwargs.get("codeql_require_ok", True), True),
        codeql_bin=str(kwargs.get("codeql_bin", "") or "").strip(),
        codeql_timeout_s=_to_int(kwargs.get("codeql_timeout_s", 120), 120),
        codeql_workdir=str(kwargs.get("codeql_workdir", "") or "").strip(),
    )

    cfg.perf_gate = _clip01(cfg.perf_gate)
    cfg.timeout_s = max(1, cfg.timeout_s)
    cfg.trace_timeout_s = max(1, cfg.trace_timeout_s)
    cfg.M_top = max(1, cfg.M_top)
    cfg.kappa = max(1e-6, cfg.kappa)

    cfg.tau_th_min = _clip01(cfg.tau_th_min)
    cfg.tau_th_max = _clip01(cfg.tau_th_max)
    cfg.tau_ac_min = _clip01(cfg.tau_ac_min)
    cfg.tau_ac_max = _clip01(cfg.tau_ac_max)
    if cfg.tau_th_max < cfg.tau_th_min: cfg.tau_th_max = cfg.tau_th_min
    if cfg.tau_ac_max < cfg.tau_ac_min: cfg.tau_ac_max = cfg.tau_ac_min

    cfg.perf_lo = _clip01(cfg.perf_lo)
    cfg.perf_hi = _clip01(cfg.perf_hi)
    if cfg.perf_hi <= cfg.perf_lo:
        cfg.perf_hi = min(1.0, cfg.perf_lo + 1e-6)

    cfg.eta_th0 = max(0.0, cfg.eta_th0)
    cfg.eta_ac0 = max(0.0, cfg.eta_ac0)
    cfg.lambda_th_max = max(0.0, cfg.lambda_th_max)
    cfg.lambda_ac_max = max(0.0, cfg.lambda_ac_max)
    cfg.ema_alpha = _clip(cfg.ema_alpha, 0.0, 1.0)
    cfg.codeql_subreward_weight = max(0.0, cfg.codeql_subreward_weight)
    cfg.codeql_subreward_scale = max(1e-6, cfg.codeql_subreward_scale)
    cfg.codeql_timeout_s = max(1, cfg.codeql_timeout_s)
    return cfg

def _compute_components(code: str, inputs: List[str], outputs: List[str], cfg: PrimalDualConfig) -> Tuple[float, float, float]:
    passed, total, _ = _run_deepcoder_eval(code, inputs, outputs, timeout_s=cfg.timeout_s, sandbox_url=cfg.sandbox_url)
    s_perf = 0.0 if total == 0 else float(passed) / float(total)

    if s_perf <= cfg.perf_gate:
        return s_perf, 0.0, 0.0

    s_thought = compute_thought_score(code, M_top=cfg.M_top, w1=cfg.w1, w2=cfg.w2) if cfg.enable_thought else 0.0
    s_action = 0.0 # Not enabled for remote sandbox test yet

    return s_perf, s_thought, s_action

def _get_pricing_context(cfg: PrimalDualConfig, fallback_perf: float) -> Tuple[float, float, float, float]:
    with _PD_LOCK:
        perf_ref = _PD_STATE.ema_perf if _PD_STATE.step > 0 else _clip01(fallback_perf)
        tau_th = _adaptive_tau(perf_ref, cfg.tau_th_min, cfg.tau_th_max, cfg.perf_lo, cfg.perf_hi)
        tau_ac = _adaptive_tau(perf_ref, cfg.tau_ac_min, cfg.tau_ac_max, cfg.perf_lo, cfg.perf_hi)
        return _PD_STATE.lambda_thought, _PD_STATE.lambda_action, tau_th, tau_ac

def _compose_reward(s_perf: float, s_thought: float, s_action: float, lambda_th: float, lambda_ac: float, tau_th: float, tau_ac: float, cfg: PrimalDualConfig) -> float:
    if s_perf <= cfg.perf_gate:
        return 0.0

    r = s_perf + lambda_th * (s_thought - tau_th) + lambda_ac * (s_action - tau_ac)
    if cfg.normalize_by_dual_mass:
        denom = 1.0 + lambda_th + lambda_ac
        if denom > 0:
            r = r / denom
    return _clip01(r)

def _update_duals(perfs: List[float], thoughts: List[float], actions: List[float], cfg: PrimalDualConfig) -> None:
    if not cfg.update_dual or len(perfs) == 0: return

    used_idx = [i for i, p in enumerate(perfs) if p > cfg.perf_gate] if cfg.dual_update_on_gated else list(range(len(perfs)))
    if len(used_idx) == 0: return

    n = float(len(used_idx))
    avg_perf = sum(perfs[i] for i in used_idx) / n
    avg_thought = sum(thoughts[i] for i in used_idx) / n
    avg_action = sum(actions[i] for i in used_idx) / n

    with _PD_LOCK:
        if _PD_STATE.step == 0:
            _PD_STATE.ema_perf = _clip01(avg_perf)
            _PD_STATE.ema_thought = _clip01(avg_thought)
            _PD_STATE.ema_action = _clip01(avg_action)
        else:
            a = cfg.ema_alpha
            _PD_STATE.ema_perf = _clip01((1.0 - a) * _PD_STATE.ema_perf + a * avg_perf)
            _PD_STATE.ema_thought = _clip01((1.0 - a) * _PD_STATE.ema_thought + a * avg_thought)
            _PD_STATE.ema_action = _clip01((1.0 - a) * _PD_STATE.ema_action + a * avg_action)

        perf_ref = _PD_STATE.ema_perf
        tau_th = _adaptive_tau(perf_ref, cfg.tau_th_min, cfg.tau_th_max, cfg.perf_lo, cfg.perf_hi)
        tau_ac = _adaptive_tau(perf_ref, cfg.tau_ac_min, cfg.tau_ac_max, cfg.perf_lo, cfg.perf_hi)

        _PD_STATE.step += 1
        eta_th = _adaptive_eta(cfg.eta_th0, perf_ref, cfg.eta_gate_center, cfg.eta_gate_scale, _PD_STATE.step)
        eta_ac = _adaptive_eta(cfg.eta_ac0, perf_ref, cfg.eta_gate_center, cfg.eta_gate_scale, _PD_STATE.step)

        _PD_STATE.lambda_thought = _clip(_PD_STATE.lambda_thought + eta_th * (tau_th - _PD_STATE.ema_thought), 0.0, cfg.lambda_th_max)
        _PD_STATE.lambda_action = _clip(_PD_STATE.lambda_action + eta_ac * (tau_ac - _PD_STATE.ema_action), 0.0, cfg.lambda_ac_max)

def compute_score_deepcoder(
    sample_or_solution: Union[Dict[str, Any], str], ground_truth: Any = None, **kwargs
) -> Union[float, Dict[str, float], List[float]]:
    cfg = _build_cfg(kwargs)
    if cfg.reset_dual_state:
        reset_primal_dual_state()

    if isinstance(sample_or_solution, dict) and ground_truth is None:
        sample = sample_or_solution
    else:
        sample = {}
        if isinstance(ground_truth, dict):
            sample.update(ground_truth)
        elif isinstance(ground_truth, str):
            sample["tests"] = ground_truth
        sample["response"] = str(sample_or_solution)

    inputs, expected_outputs = _get_tests_deepcoder(sample)

    def _score_many_with_info(responses: List[str]) -> List[Dict[str, float]]:
        if len(responses) == 0:
            return []

        perfs = []
        thoughts = []
        actions = []
        codes = []
        for resp in responses:
            code = _extract_code(str(resp))
            codes.append(code)
            s_perf, s_thought, s_action = _compute_components(code, inputs, expected_outputs, cfg)
            perfs.append(s_perf)
            thoughts.append(s_thought)
            actions.append(s_action)

        batch_perf = sum(perfs) / float(len(perfs))
        lambda_th, lambda_ac, tau_th, tau_ac = _get_pricing_context(cfg, fallback_perf=batch_perf)

        rewards = [
            _compose_reward(sp, st, sa, lambda_th, lambda_ac, tau_th, tau_ac, cfg)
            for sp, st, sa in zip(perfs, thoughts, actions)
        ]

        codeql_scalars = [0.0] * len(rewards)
        codeql_robust_scores = [0.0] * len(rewards)
        codeql_subrewards = [0.0] * len(rewards)
        codeql_nodes = [0] * len(rewards)
        codeql_edges = [0] * len(rewards)
        codeql_densities = [0.0] * len(rewards)
        codeql_oks = [0.0] * len(rewards)

        if cfg.enable_codeql_subreward:
            for i, (sp, code) in enumerate(zip(perfs, codes)):
                # Only add structural subreward when unit tests are fully passed.
                if not (sp >= 1.0 - 1e-12):
                    continue
                codeql_res = compute_codeql_robustness(
                    code=code,
                    codeql_bin=cfg.codeql_bin,
                    timeout_s=cfg.codeql_timeout_s,
                    workdir=cfg.codeql_workdir or None,
                )
                codeql_oks[i] = 1.0 if codeql_res.ok else 0.0
                codeql_scalars[i] = float(codeql_res.scalar)
                codeql_nodes[i] = int(codeql_res.num_nodes)
                codeql_edges[i] = int(codeql_res.num_edges)
                codeql_densities[i] = float(codeql_res.density)

                if codeql_res.ok or not cfg.codeql_require_ok:
                    robust_score = normalize_codeql_scalar(
                        codeql_res.scalar,
                        threshold=cfg.codeql_subreward_threshold,
                        scale=cfg.codeql_subreward_scale,
                    )
                    bonus = cfg.codeql_subreward_weight * robust_score
                    rewards[i] = _clip01(rewards[i] + bonus)
                    codeql_robust_scores[i] = float(robust_score)
                    codeql_subrewards[i] = float(bonus)

        _update_duals(perfs, thoughts, actions, cfg)
        state = get_primal_dual_state()

        infos = []
        for i, (sp, st, sa, reward) in enumerate(zip(perfs, thoughts, actions, rewards)):
            infos.append(
                {
                    "score": float(reward),
                    "combined_reward": float(reward),
                    "acc": float(sp),
                    "original_reward": float(sp),
                    "thought_reward": float(st),
                    "action_reward": float(sa),
                    "codeql_subreward": float(codeql_subrewards[i]),
                    "codeql_robust_score": float(codeql_robust_scores[i]),
                    "codeql_robust_scalar": float(codeql_scalars[i]),
                    "codeql_subreward_weight": float(cfg.codeql_subreward_weight),
                    "codeql_subreward_applied": float(1.0 if codeql_subrewards[i] > 0.0 else 0.0),
                    "codeql_ok": float(codeql_oks[i]),
                    "codeql_nodes": float(codeql_nodes[i]),
                    "codeql_edges": float(codeql_edges[i]),
                    "codeql_density": float(codeql_densities[i]),
                    "lambda_thought": float(lambda_th),
                    "lambda_action": float(lambda_ac),
                    "tau_thought": float(tau_th),
                    "tau_action": float(tau_ac),
                    "ema_perf": float(state["ema_perf"]),
                    "ema_thought": float(state["ema_thought"]),
                    "ema_action": float(state["ema_action"]),
                    "dual_step": float(state["step"]),
                }
            )
        return infos

    if isinstance(sample.get("responses", None), list):
        return [info["score"] for info in _score_many_with_info(sample["responses"])]

    for k in ("response", "completion", "output", "generated_text", "text"):
        if isinstance(sample.get(k, None), str) and sample[k].strip():
            return _score_many_with_info([sample[k]])[0]

    return 0.0
