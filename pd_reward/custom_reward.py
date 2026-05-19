import os
import sys
from typing import Any

# Add current directory to sys.path to allow importing local reward_score package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo

from reward_score.primal_dual_core import GenericRewardCombiner
from reward_score.sub_reward import collect_subrewards, weight_overrides, to_bool, clip
import reward_score.mbpp_action_thought_reward as mbpp_evaluator
import reward_score.deepcoder_action_thought_reward as deepcoder_evaluator

# Register PDAR advantage estimator when this module is imported
try:
    import pdar_init  # noqa: F401 — triggers @register_adv_est("pdar")
except ImportError:
    pass  # PDAR not available in this installation

# Initialize the generic combiner (will parse kwargs for combine_mode="pd"|"multiplier")
# We delay initialization until the first call to ensure we have the kwargs


MATH_DATA_SOURCES = {
    "general365",
    "openr1_math_220k",
    "deepscalar",
    "math_dapo",
    "math",
    "math_dapo_reasoning",
    "openai/gsm8k",
    "gsm8k",
    "amc",
    "aops",
    "cn_k12",
    "math500",
    "numina",
    "synthetic",
    "olympiad",
}


_COMBINER_CACHE = {}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extra_info_dict(extra_info: Any) -> dict[str, Any]:
    return extra_info if isinstance(extra_info, dict) else {}


def _global_step(extra_info: Any) -> int:
    extra = _extra_info_dict(extra_info)
    for key in ("global_step", "global_steps"):
        if key in extra and extra[key] is not None:
            try:
                return int(extra[key])
            except (TypeError, ValueError):
                return -1
    return -1


def _should_update_dual(extra_info: Any) -> bool:
    return not to_bool(_extra_info_dict(extra_info).get("is_validation", False), False)


def _subreward_weight(name: str, kwargs: dict[str, Any]) -> float:
    candidates = [f"weight_{name}"]
    for category in ("math", "coding"):
        prefix = f"{category}_"
        if name.startswith(prefix):
            candidates.append(f"{category}_weight_{name[len(prefix):]}")
    for key in candidates:
        if key in kwargs:
            return _to_float(kwargs[key], 1.0)
    return 1.0


def _weighted_subreward_average(subrewards: dict[str, float], kwargs: dict[str, Any]) -> float:
    if not subrewards:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, value in subrewards.items():
        weight = _subreward_weight(name, kwargs)
        if weight <= 0:
            continue
        weighted_sum += weight * float(value)
        weight_sum += weight
    if weight_sum <= 0:
        return sum(float(v) for v in subrewards.values()) / max(len(subrewards), 1)
    return weighted_sum / weight_sum


def _pdar_reward_info(
    main_reward: float,
    subrewards: dict[str, float],
    kwargs: dict[str, Any],
    *,
    signed: bool = False,
) -> dict[str, Any]:
    main = float(main_reward)
    if signed:
        main = 1.0 if main > 0.0 else -1.0
    return {
        "score": main,
        "main_reward": main,
        "aux_reward_combined": _weighted_subreward_average(subrewards, kwargs),
        "aux_rewards": subrewards,
        "acc": bool(main_reward > 0.0),
        "original_reward": float(main_reward),
    }


def _get_combiner(kwargs: dict[str, Any]) -> GenericRewardCombiner:

    combine_mode = str(kwargs.get("combine_mode", "none")).lower()
    if combine_mode not in _COMBINER_CACHE:
        combiner_kwargs = {k: v for k, v in kwargs.items() if k not in ["combine_mode"]}
        combiner_kwargs.update(weight_overrides("coding", **kwargs))
        combiner_kwargs.update(weight_overrides("math", **kwargs))
        _COMBINER_CACHE[combine_mode] = GenericRewardCombiner(combine_mode=combine_mode, subreward_names=[], **combiner_kwargs)
    
    return _COMBINER_CACHE[combine_mode]


def _score_math(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in {"openai/gsm8k", "gsm8k"}:
        from verl.utils.reward_score import gsm8k

        base_res = gsm8k.compute_score(solution_str, ground_truth)
        
        # Fallback for models that output \boxed{} instead of ####
        if isinstance(base_res, (int, float)) and base_res == 0.0 or (isinstance(base_res, dict) and base_res.get("score", 0.0) == 0.0):
            if "\\boxed{" in solution_str:
                base_res = math_dapo.compute_score(solution_str, ground_truth)
            else:
                base_res = gsm8k.compute_score(solution_str, ground_truth, method="flexible")
                
        if isinstance(base_res, dict):
            base_score = float(base_res.get("score", base_res.get("acc", 0.0)))
            base_acc = bool(base_res.get("acc", base_score > 0.0))
        else:
            base_score = float(base_res)
            base_acc = base_score > 0.0
            base_res = {"score": base_score, "acc": base_acc}
    else:
        base_res = math_dapo.compute_score(solution_str, ground_truth)
        base_score = float(base_res.get("score", 0.0)) if isinstance(base_res, dict) else float(base_res)
        base_acc = bool(base_res.get("acc", base_score > 0.0)) if isinstance(base_res, dict) else base_score > 0.0

    combine_mode = str(kwargs.get("combine_mode", "none")).lower()
    if combine_mode == "none" or not to_bool(kwargs.get("math_enable_sub_rewards", False), False):
        if to_bool(kwargs.get("math_signed_reward", True), True):
            signed_score = 1.0 if base_acc else -1.0
            if isinstance(base_res, dict):
                base_res["score"] = signed_score
            else:
                base_res = signed_score
        return base_res

    ctx = {
        "response": solution_str,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info or {},
        "data_source": data_source,
        "base_score": base_score,
        "base_acc": base_acc,
    }
    subrewards = collect_subrewards("math", ctx, **kwargs)
    if not subrewards:
        if to_bool(kwargs.get("math_signed_reward", True), True):
            signed_score = 1.0 if base_acc else -1.0
            if isinstance(base_res, dict):
                base_res["score"] = signed_score
            else:
                base_res = signed_score
        return base_res

    # --- PDAR mode: return separate channels, skip reward-level combination ---
    if combine_mode == "pdar":
        info = _pdar_reward_info(
            1.0 if base_acc else 0.0,
            subrewards,
            kwargs,
            signed=to_bool(kwargs.get("math_signed_reward", True), True),
        )
        info["base_math_score"] = float(base_score)
        info["original_reward"] = float(base_score)
        return info

    combiner = _get_combiner(kwargs)
    main_reward = 1.0 if base_acc else 0.0
    info = combiner.process_batch(
        [main_reward],
        [subrewards],
        global_step=_global_step(extra_info),
        update_dual=_should_update_dual(extra_info),
    )[0]

    if to_bool(kwargs.get("math_signed_reward", True), True):
        if combine_mode == "pd":
            # PD rewards are already in [-1, 1] from process_batch clipping;
            # remap to signed scale: positive for correct, negative for wrong
            info["score"] = clip(info["score"], -1.0, 1.0)
        else:
            info["score"] = 2.0 * clip(info["score"], 0.0, 1.0) - 1.0
        info["combined_reward"] = info["score"]

    info["base_math_score"] = float(base_score)
    info["original_reward"] = float(base_score)
    info["acc"] = bool(base_acc)
    if isinstance(base_res, dict) and "pred" in base_res:
        info["pred"] = base_res["pred"]
    return info

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ds_str = str(data_source)
    if ds_str in MATH_DATA_SOURCES or ds_str.startswith("aime") or ds_str.startswith("deepscalar") or "math" in ds_str:
        return _score_math(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)

    combine_mode = str(kwargs.get("combine_mode", "none")).lower()
    combiner = _get_combiner(kwargs)
        
    if data_source in ["mbpp:train", "mbpp:test", "mbpp:validation", "mbpp"]:
        # Pass return_components=True to just get the raw components
        res = mbpp_evaluator.compute_score_mbpp(solution_str, ground_truth, return_components=True, **kwargs)
        
        # Determine if it's a batch or single
        if not isinstance(res, (list, dict)):
            # Fallback: tuple (main_reward, subrewards)
            main_reward, subrewards = res
            res = {"main_reward": main_reward, "subrewards": subrewards}

        if combine_mode == "pdar":
            if isinstance(res, list):
                return [
                    _pdar_reward_info(r["main_reward"], r["subrewards"], kwargs, signed=False)
                    for r in res
                ]
            return _pdar_reward_info(res["main_reward"], res["subrewards"], kwargs, signed=False)

        if isinstance(res, list):
            main_rewards = [r["main_reward"] for r in res]
            subrewards_list = [r["subrewards"] for r in res]
            infos = combiner.process_batch(
                main_rewards,
                subrewards_list,
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return [info["score"] for info in infos]
        else:
            infos = combiner.process_batch(
                [res["main_reward"]],
                [res["subrewards"]],
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return infos[0] if infos else 0.0
            
    elif data_source.startswith("deepcoder"):
        sandbox_fusion_url = kwargs.get("sandbox_fusion_url", None)
        concurrent_semaphore = kwargs.get("concurrent_semaphore", None)

        res = deepcoder_evaluator.compute_score_deepcoder(
            solution_str, ground_truth, 
            sandbox_url=sandbox_fusion_url, 
            concurrent_semaphore=concurrent_semaphore, 
            return_components=True, 
            **kwargs
        )
        
        if not isinstance(res, (list, dict)):
            # Fallback: tuple (main_reward, subrewards)
            main_reward, subrewards = res
            res = {"main_reward": main_reward, "subrewards": subrewards}

        if combine_mode == "pdar":
            if isinstance(res, list):
                return [
                    _pdar_reward_info(r["main_reward"], r["subrewards"], kwargs, signed=False)
                    for r in res
                ]
            return _pdar_reward_info(res["main_reward"], res["subrewards"], kwargs, signed=False)

        if isinstance(res, list):
            main_rewards = [r["main_reward"] for r in res]
            subrewards_list = [r["subrewards"] for r in res]
            infos = combiner.process_batch(
                main_rewards,
                subrewards_list,
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return [info["score"] for info in infos]
        else:
            infos = combiner.process_batch(
                [res["main_reward"]],
                [res["subrewards"]],
                global_step=_global_step(extra_info),
                update_dual=_should_update_dual(extra_info),
            )
            return infos[0] if infos else 0.0

    # Fallback to the default compute score for other data sources
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
