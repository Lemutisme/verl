import math
from threading import Lock
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

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

class GenericPrimalDualState:
    def __init__(self):
        self.lambdas: Dict[str, float] = {}
        self.ema_perf: float = 0.0
        self.ema_subrewards: Dict[str, float] = {}
        self.step: int = 0
        
    def asdict(self):
        d = {"ema_perf": self.ema_perf, "step": self.step}
        for k, v in self.lambdas.items():
            d[f"lambda_{k}"] = v
        for k, v in self.ema_subrewards.items():
            d[f"ema_{k}"] = v
        return d

@dataclass
class SubrewardConfig:
    tau_min: float = 0.20
    tau_max: float = 0.85
    eta0: float = 0.05
    lambda_max: float = 4.0
    static_multiplier: float = 1.0

# Default warmup steps: use higher EMA alpha for faster convergence during initial training
_DEFAULT_WARMUP_STEPS: int = 10
_DEFAULT_WARMUP_ALPHA: float = 0.30

class GenericRewardCombiner:
    """
    A generic combiner that merges a main reward (acc) with arbitrary subrewards
    using either a static multiplier or dynamic primal-dual adaptive lambdas.

    NOTE: State (_state) is per-instance so that separate combiner instances
    (e.g. coding vs math) maintain independent EMA and lambda values.
    """

    def __init__(self, combine_mode: str = "multiplier", subreward_names: List[str] = None, **kwargs):
        """
        combine_mode: 'pd' (primal-dual) or 'multiplier' (static weights) or 'none' (only main reward)
        """
        self.combine_mode = str(combine_mode).strip().lower()
        if self.combine_mode in {"new", "static"}:
            self.combine_mode = "multiplier"
            
        self.subreward_names = subreward_names or []
        
        self.perf_gate = _to_float(kwargs.get("perf_gate", -1.0), -1.0) # Default to -1.0 to allow sub-rewards when acc is 0
        self.perf_lo = _clip01(_to_float(kwargs.get("perf_lo", 0.20), 0.20))
        self.perf_hi = _clip01(_to_float(kwargs.get("perf_hi", 0.90), 0.90))
        if self.perf_hi <= self.perf_lo:
            self.perf_hi = min(1.0, self.perf_lo + 1e-6)
            
        self.eta_gate_center = _to_float(kwargs.get("eta_gate_center", 0.40), 0.40)
        self.eta_gate_scale = _to_float(kwargs.get("eta_gate_scale", 8.0), 8.0)
        self.ema_alpha = _clip(_to_float(kwargs.get("ema_alpha", 0.05), 0.05), 0.0, 1.0)
        self.warmup_steps = _to_int(kwargs.get("warmup_steps", _DEFAULT_WARMUP_STEPS), _DEFAULT_WARMUP_STEPS)
        self.warmup_alpha = _clip(_to_float(kwargs.get("warmup_alpha", _DEFAULT_WARMUP_ALPHA), _DEFAULT_WARMUP_ALPHA), 0.0, 1.0)
        self.update_dual = _to_bool(kwargs.get("update_dual", True), True)
        self.dual_update_on_gated = _to_bool(kwargs.get("dual_update_on_gated", False), False)
        self.normalize_by_dual_mass = _to_bool(kwargs.get("normalize_by_dual_mass", False), False)
        
        self.kwargs = kwargs
        
        # Instance-level state and lock (not shared across instances)
        self._lock = Lock()
        self._state = GenericPrimalDualState()
        
        self.sub_cfgs: Dict[str, SubrewardConfig] = {}
        for name in self.subreward_names:
            self._register_subreward(name)

        if _to_bool(kwargs.get("reset_dual_state", False), False):
            self.reset_state()

    def _register_subreward(self, name: str):
        if name in self.sub_cfgs:
            return
            
        self.sub_cfgs[name] = SubrewardConfig(
            tau_min=_clip01(_to_float(self.kwargs.get(f"tau_{name}_min", 0.20), 0.20)),
            tau_max=_clip01(_to_float(self.kwargs.get(f"tau_{name}_max", 0.85), 0.85)),
            eta0=max(0.0, _to_float(self.kwargs.get(f"eta_{name}", 0.05), 0.05)),
            lambda_max=max(0.0, _to_float(self.kwargs.get(f"lambda_{name}_max", 4.0), 4.0)),
            static_multiplier=_to_float(self.kwargs.get(f"weight_{name}", 1.0), 1.0) # default weight 1.0
        )
        # Ensure tau_max >= tau_min
        if self.sub_cfgs[name].tau_max < self.sub_cfgs[name].tau_min:
            self.sub_cfgs[name].tau_max = self.sub_cfgs[name].tau_min
            
    def _ensure_dynamic_registration(self, subrewards_list: List[Dict[str, float]]):
        if not subrewards_list:
            return
            
        new_keys = set()
        for d in subrewards_list:
            new_keys.update(d.keys())
            
        for name in sorted(list(new_keys)):
            if name not in self.subreward_names:
                with self._lock:
                    if name not in self.subreward_names:
                        self.subreward_names.append(name)
                        self._register_subreward(name)
                        # Initialize state for new subreward if we are already mid-training
                        if name not in self._state.lambdas:
                            self._state.lambdas[name] = 0.0
                        if name not in self._state.ema_subrewards:
                            self._state.ema_subrewards[name] = 0.0

    def reset_state(self):
        with self._lock:
            self._state = GenericPrimalDualState()

    def _get_pricing_context(self, fallback_perf: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        with self._lock:
            perf_ref = self._state.ema_perf if self._state.step > 0 else _clip01(fallback_perf)
            lambdas = {}
            taus = {}
            for name, cfg in self.sub_cfgs.items():
                lambdas[name] = self._state.lambdas.get(name, 0.0)
                taus[name] = _adaptive_tau(perf_ref, cfg.tau_min, cfg.tau_max, self.perf_lo, self.perf_hi)
            return lambdas, taus

    def _update_duals(self, perfs: List[float], subrewards_list: List[Dict[str, float]]):
        if not self.update_dual or len(perfs) == 0:
            return

        used_idx = [i for i, p in enumerate(perfs) if p > self.perf_gate] if self.dual_update_on_gated else list(range(len(perfs)))
        if len(used_idx) == 0:
            return

        n = float(len(used_idx))
        avg_perf = sum(perfs[i] for i in used_idx) / n
        avg_subs = {}
        for name in self.subreward_names:
            avg_subs[name] = sum(subrewards_list[i].get(name, 0.0) for i in used_idx) / n

        with self._lock:
            # Use higher alpha during warmup for faster convergence
            a = self.warmup_alpha if self._state.step < self.warmup_steps else self.ema_alpha

            if self._state.step == 0:
                self._state.ema_perf = _clip01(avg_perf)
                for name in self.subreward_names:
                    self._state.ema_subrewards[name] = _clip01(avg_subs[name])
                    if name not in self._state.lambdas:
                        self._state.lambdas[name] = 0.0
            else:
                self._state.ema_perf = _clip01((1.0 - a) * self._state.ema_perf + a * avg_perf)
                for name in self.subreward_names:
                    self._state.ema_subrewards[name] = _clip01((1.0 - a) * self._state.ema_subrewards.get(name, 0.0) + a * avg_subs[name])

            perf_ref = self._state.ema_perf
            self._state.step += 1

            for name, cfg in self.sub_cfgs.items():
                tau_name = _adaptive_tau(perf_ref, cfg.tau_min, cfg.tau_max, self.perf_lo, self.perf_hi)
                eta_name = _adaptive_eta(cfg.eta0, perf_ref, self.eta_gate_center, self.eta_gate_scale, self._state.step)
                
                curr_lambda = self._state.lambdas.get(name, 0.0)
                curr_ema_sub = self._state.ema_subrewards.get(name, 0.0)
                
                self._state.lambdas[name] = _clip(curr_lambda + eta_name * (tau_name - curr_ema_sub), 0.0, cfg.lambda_max)

    def process_batch(self, main_rewards: List[float], subrewards_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Processes a batch of responses and calculates the combined reward + infos.
        """
        if len(main_rewards) == 0:
            return []
            
        self._ensure_dynamic_registration(subrewards_list)

        if self.combine_mode == "none":
            # Just return main rewards
            return [{"score": float(r), "combined_reward": float(r), "acc": float(r)} for r in main_rewards]

        infos = []
        batch_perf = sum(main_rewards) / float(len(main_rewards))
        
        if self.combine_mode == "pd":
            lambdas, taus = self._get_pricing_context(fallback_perf=batch_perf)
            self._update_duals(main_rewards, subrewards_list)
            
            with self._lock:
                state_dict = self._state.asdict()
                
            for idx, s_perf in enumerate(main_rewards):
                # Only apply perf_gate if it's explicitly positive
                if self.perf_gate > 0 and s_perf < self.perf_gate:
                    reward = 0.0
                else:
                    reward = s_perf
                    for name in self.subreward_names:
                        s_sub = subrewards_list[idx].get(name, 0.0)
                        reward += lambdas[name] * (s_sub - taus[name])
                    
                    if self.normalize_by_dual_mass:
                        denom = 1.0 + sum(lambdas.values())
                        if denom > 0:
                            reward = reward / denom
                            
                # PD rewards can be negative (penalty signal); clip to [-1, 1]
                reward = _clip(reward, -1.0, 1.0)
                info = {
                    "score": float(reward),
                    "combined_reward": float(reward),
                    "acc": float(s_perf),
                    "original_reward": float(s_perf)
                }
                for name in self.subreward_names:
                    s_sub = subrewards_list[idx].get(name, 0.0)
                    info[f"{name}_reward"] = float(s_sub)
                    info[f"lambda_{name}"] = float(lambdas[name])
                    info[f"tau_{name}"] = float(taus[name])
                    
                info.update({k: float(v) for k, v in state_dict.items()})
                infos.append(info)
                
        elif self.combine_mode == "multiplier":
            for idx, s_perf in enumerate(main_rewards):
                if self.perf_gate > 0 and s_perf < self.perf_gate:
                    reward = 0.0
                else:
                    reward = s_perf
                    for name, cfg in self.sub_cfgs.items():
                        s_sub = subrewards_list[idx].get(name, 0.0)
                        reward += cfg.static_multiplier * s_sub

                reward = _clip01(reward)
                info = {
                    "score": float(reward),
                    "combined_reward": float(reward),
                    "acc": float(s_perf),
                    "original_reward": float(s_perf)
                }
                for name in self.subreward_names:
                    s_sub = subrewards_list[idx].get(name, 0.0)
                    info[f"{name}_reward"] = float(s_sub)
                    info[f"weight_{name}"] = float(self.sub_cfgs[name].static_multiplier)
                infos.append(info)

        return infos
