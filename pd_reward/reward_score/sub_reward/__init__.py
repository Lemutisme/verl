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
        "code_extractability_reward": True,
        "syntax_validity_reward": True,
        "unit_test_pass_rate": False,
        "compiler_runtime_feedback": True,
        "static_analysis_reward": False,
        "executed_token_credit": False,
        "block_level_process_reward": False,
    },
    "math": {
        "final_answer_reward": False,  # Disabled: redundant with main accuracy reward in PD mode
        "answer_efficiency_reward": False,
        "consistency_reward": False,
        "executable_unit_pass_rate_reward": False,
        "step_arithmetic_validity_reward": True,
        "prefix_consistency_reward": True,
        "trace_efficiency_reward": True,
        "answer_extractability_reward": True,
    },
}

DEFAULT_WEIGHTS = {
    "coding": {
        "code_extractability_reward": 0.15,
        "syntax_validity_reward": 0.25,
        "unit_test_pass_rate": 0.0,
        "compiler_runtime_feedback": 0.30,
        "static_analysis_reward": 0.0,
        "executed_token_credit": 0.0,
        "block_level_process_reward": 0.0,
    },
    "math": {
        "final_answer_reward": 0.0,
        "answer_efficiency_reward": 0.0,
        "consistency_reward": 0.0,
        "executable_unit_pass_rate_reward": 0.0,
        "step_arithmetic_validity_reward": 0.35,
        "prefix_consistency_reward": 0.15,
        "trace_efficiency_reward": 0.35,
        "answer_extractability_reward": 0.15,
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
