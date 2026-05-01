from typing import Any

from .common import clip, to_bool, to_float
from .coding import MODULES as CODING_MODULES
from .math import MODULES as MATH_MODULES


CATEGORY_MODULES = {
    "coding": CODING_MODULES,
    "math": MATH_MODULES,
}

DEFAULT_ENABLED = {
    "coding": {
        "unit_test_pass_rate": False,
        "compiler_runtime_feedback": True,
        "static_analysis_reward": True,
        "executed_token_credit": True,
        "block_level_process_reward": True,
    },
    "math": {
        "final_answer_reward": False,
        "boxed_format_reward": True,
        "step_reachability_reward": True,
        "progress_reward": True,
        "reasoning_quality_heuristic": True,
        "reasoning_distance_reward": True,
    },
}

DEFAULT_WEIGHTS = {
    "coding": {
        "unit_test_pass_rate": 0.0,
        "compiler_runtime_feedback": 0.20,
        "static_analysis_reward": 0.15,
        "executed_token_credit": 0.10,
        "block_level_process_reward": 0.10,
    },
    "math": {
        "final_answer_reward": 0.0,
        "boxed_format_reward": 0.10,
        "step_reachability_reward": 0.10,
        "progress_reward": 0.10,
        "reasoning_quality_heuristic": 0.10,
        "reasoning_distance_reward": 0.08,
    },
}


def _prefixed_name(category: str, name: str) -> str:
    return f"{category}_{name}"


def _get_bool(kwargs: dict[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key in kwargs:
            return to_bool(kwargs[key], default)
    return default


def _get_float(kwargs: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in kwargs:
            return to_float(kwargs[key], default)
    return default


def category_enabled(category: str, kwargs: dict[str, Any]) -> bool:
    return _get_bool(
        kwargs,
        f"{category}_enable_sub_rewards",
        "enable_sub_rewards",
        default=False,
    )


def collect_subrewards(category: str, ctx: dict[str, Any], **kwargs: Any) -> dict[str, float]:
    modules = CATEGORY_MODULES.get(category, {})
    if not modules or not category_enabled(category, kwargs):
        return {}

    subrewards: dict[str, float] = {}
    for name, module in modules.items():
        default_enabled = DEFAULT_ENABLED.get(category, {}).get(name, True)
        enabled = _get_bool(
            kwargs,
            f"{category}_enable_{name}",
            f"enable_{_prefixed_name(category, name)}",
            f"enable_{name}",
            default=default_enabled,
        )
        if not enabled:
            continue

        raw_score = module.compute(ctx, **kwargs)
        score = clip(raw_score)
        weight = _get_float(
            kwargs,
            f"{category}_weight_{name}",
            f"weight_{_prefixed_name(category, name)}",
            default=DEFAULT_WEIGHTS.get(category, {}).get(name, 1.0),
        )

        sub_name = _prefixed_name(category, name)
        subrewards[sub_name] = score
        kwargs.setdefault(f"weight_{sub_name}", weight)
    return subrewards


def weight_overrides(category: str, **kwargs: Any) -> dict[str, float]:
    weights = {}
    for name in CATEGORY_MODULES.get(category, {}):
        sub_name = _prefixed_name(category, name)
        weights[f"weight_{sub_name}"] = _get_float(
            kwargs,
            f"{category}_weight_{name}",
            f"weight_{sub_name}",
            default=DEFAULT_WEIGHTS.get(category, {}).get(name, 1.0),
        )
    return weights
