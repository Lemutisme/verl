import os
import sys

# Add current directory to sys.path to allow importing local reward_score package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo

from reward_score.primal_dual_core import GenericRewardCombiner
import reward_score.mbpp_action_thought_reward as mbpp_evaluator
import reward_score.deepcoder_action_thought_reward as deepcoder_evaluator

# Initialize the generic combiner (will parse kwargs for combine_mode="pd"|"multiplier")
# We delay initialization until the first call to ensure we have the kwargs
_combiner = None

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    global _combiner
    
    if data_source in ["general365", "openr1_math_220k", "deepscalar"]:
        res = math_dapo.compute_score(solution_str, ground_truth)
        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
            
    if _combiner is None:
        combine_mode = kwargs.get("combine_mode", "pd")
        
        # Filter out combine_mode from kwargs to avoid duplicate keyword argument error
        combiner_kwargs = {k: v for k, v in kwargs.items() if k != "combine_mode"}
        _combiner = GenericRewardCombiner(combine_mode=combine_mode, subreward_names=[], **combiner_kwargs)
        
    if data_source in ["mbpp:train", "mbpp:test", "mbpp:validation", "mbpp"]:
        # Pass return_components=True to just get the raw components
        res = mbpp_evaluator.compute_score_mbpp(solution_str, ground_truth, return_components=True, **kwargs)
        
        # Determine if it's a batch or single
        if isinstance(res, list):
            main_rewards = [r["main_reward"] for r in res]
            subrewards_list = [r["subrewards"] for r in res]
            infos = _combiner.process_batch(main_rewards, subrewards_list)
            return [info["score"] for info in infos]
        else:
            main_reward, subrewards = res
            infos = _combiner.process_batch([main_reward], [subrewards])
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
            infos = _combiner.process_batch(main_rewards, subrewards_list)
            return [info["score"] for info in infos]
        else:
            main_reward, subrewards = res
            infos = _combiner.process_batch([main_reward], [subrewards])
            return infos[0] if infos else 0.0

    # Fallback to the default compute score for other data sources
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)

