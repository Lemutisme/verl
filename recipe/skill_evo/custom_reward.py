import os
import sys

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

# Initialize the generic combiner (will parse kwargs for combine_mode="pd"|"multiplier")
# We delay initialization until the first call to ensure we have the kwargs
_combiner = None


MATH_DATA_SOURCES = {
    "general365",
    "openr1_math_220k",
    "deepscalar",
    "math_dapo",
    "math",
    "math_dapo_reasoning",
    "openai/gsm8k",
    "gsm8k",
}


_COMBINER_CACHE = {}

def _get_combiner(kwargs: dict[str, Any]) -> GenericRewardCombiner:
    print(f"DEBUG: _get_combiner called with kwargs={kwargs}")
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
        print(f"DEBUG bypass: combine_mode={combine_mode}, math_enable_sub_rewards={kwargs.get('math_enable_sub_rewards')}")
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
        return base_res

    combiner = _get_combiner(kwargs)
    main_reward = 1.0 if base_acc else 0.0
    info = combiner.process_batch([main_reward], [subrewards])[0]

    if to_bool(kwargs.get("math_signed_reward", True), True):
        info["score"] = 2.0 * clip(info["score"]) - 1.0
        info["combined_reward"] = info["score"]

    info["base_math_score"] = float(base_score)
    info["original_reward"] = float(base_score)
    info["acc"] = bool(base_acc)
    if isinstance(base_res, dict) and "pred" in base_res:
        info["pred"] = base_res["pred"]
    return info

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in MATH_DATA_SOURCES or data_source.startswith("aime"):
        return _score_math(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)

    combiner = _get_combiner(kwargs)
        
    if data_source in ["mbpp:train", "mbpp:test", "mbpp:validation", "mbpp"]:
        # Pass return_components=True to just get the raw components
        res = mbpp_evaluator.compute_score_mbpp(solution_str, ground_truth, return_components=True, **kwargs)
        
        # Determine if it's a batch or single
        if isinstance(res, list):
            main_rewards = [r["main_reward"] for r in res]
            subrewards_list = [r["subrewards"] for r in res]
            infos = combiner.process_batch(main_rewards, subrewards_list)
            return [info["score"] for info in infos]
        else:
            main_reward, subrewards = res
            infos = combiner.process_batch([main_reward], [subrewards])
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
        
        if isinstance(res, list):
            main_rewards = [r["main_reward"] for r in res]
            subrewards_list = [r["subrewards"] for r in res]
            infos = combiner.process_batch(main_rewards, subrewards_list)
            return [info["score"] for info in infos]
        else:
            main_reward, subrewards = res
            infos = combiner.process_batch([main_reward], [subrewards])
            return infos[0] if infos else 0.0

    # Fallback to the default compute score for other data sources
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
