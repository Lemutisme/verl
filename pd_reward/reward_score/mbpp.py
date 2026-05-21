from typing import Any

from reward_score.coding_executable_reward import compute_score_coding


def compute_score_mbpp(sample_or_solution: dict | str, ground_truth: Any = None, **kwargs):
    return compute_score_coding(sample_or_solution, ground_truth, eval_mode="assert", **kwargs)
