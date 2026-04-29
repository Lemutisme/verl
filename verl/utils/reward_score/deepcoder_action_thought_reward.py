import os
import re
import json
import threading
from typing import Any, Dict, List, Tuple, Union
import ast
import math
from collections import defaultdict

# Reuse the sandbox execution wrapper from sandbox_fusion
# (Assuming sandbox_fusion/utils.py is available in verl.utils.reward_score)
from verl.utils.reward_score.sandbox_fusion.utils import check_correctness, DEFAULT_TIMEOUT

_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def _extract_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Try finding closed block first
    m = _CODEBLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # If no closed block is found, try finding an unclosed block
    m_unclosed = re.search(r"```(?:python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m_unclosed:
        return m_unclosed.group(1).strip()
    return text.strip()

def _get_tests_deepcoder(sample: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    """
    Extract inputs and outputs pairs from DeepCoder tests JSON string.
    """
    tests_str = sample.get("tests", "{}")
    if not tests_str:
        return [], []
    try:
        data = json.loads(tests_str)
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        return inputs, outputs
    except Exception:
        return [], []

def _ast_depth_stats(tree: ast.AST):
    depth_cnt = defaultdict(int)
    max_depth = 0

    def dfs(node, d: int):
        nonlocal max_depth
        depth_cnt[d] += 1
        if d > max_depth:
            max_depth = d
        for ch in ast.iter_child_nodes(node):
            dfs(ch, d + 1)

    dfs(tree, 0)
    total = sum(depth_cnt.values())
    return depth_cnt, total, max_depth

def compute_thought_score(code: str, M_top: int = 25, w1: float = 0.7, w2: float = 0.3) -> float:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0

    depth_cnt, total, K = _ast_depth_stats(tree)
    if total <= 1 or K <= 0:
        S_depth = 1.0
    else:
        H = 0.0
        for cnt in depth_cnt.values():
            p = cnt / total
            if p > 0:
                H -= p * math.log(p)
        denom = math.log(K + 1)
        H = 0.0 if denom <= 0 else H / denom
        if H < 0.0:
            H = 0.0
        if H > 1.0:
            H = 1.0
        S_depth = 1.0 - H

    top_body = getattr(tree, "body", None)
    F_top = len(top_body) if isinstance(top_body, list) else 0
    S_top = 1.0 - (min(F_top, M_top) / float(M_top))

    S = w1 * S_depth + w2 * S_top
    if S < 0.0:
        S = 0.0
    if S > 1.0:
        S = 1.0
    return float(S)

def compute_action_score_from_sums(revisit_sum: float, cost_sum: float, nt: int, u1: float = 0.5, u2: float = 0.5, kappa: float = 8.0) -> float:
    if nt <= 0:
        return 0.0

    D_revisit = revisit_sum / float(nt)
    if D_revisit < 0.0:
        D_revisit = 0.0
    if D_revisit > 1.0:
        D_revisit = 1.0
    S_revisit = 1.0 - D_revisit

    D_cost = cost_sum / float(nt)
    if D_cost < 0.0:
        D_cost = 0.0
    if kappa <= 0:
        S_cost = 0.0
    else:
        S_cost = math.exp(-D_cost / kappa)

    S = u1 * S_revisit + u2 * S_cost
    if S < 0.0:
        S = 0.0
    if S > 1.0:
        S = 1.0
    return float(S)

def combine_reward(S_perf: float, S_thought: float, S_action: float, beta: float = 1.0, gamma: float = 1.0) -> float:
    denom = 1.0 + beta + gamma
    if denom <= 0:
        r = S_perf
    else:
        r = (S_perf * (1.0 + beta * S_thought + gamma * S_action)) / denom
    if r < 0.0:
        r = 0.0
    if r > 1.0:
        r = 1.0
    return float(r)

def _run_deepcoder_eval(code: str, inputs: List[str], expected_outputs: List[str], timeout_s: int = 10, sandbox_url: str = "http://localhost:8000/run", concurrent_semaphore=None) -> Tuple[int, int, str]:
    if not inputs:
        return 0, 0, "no_tests"

    in_outs = {"inputs": inputs, "outputs": expected_outputs}
    # Using check_correctness from sandbox_fusion. 
    # Returns results list: True, False, -1 (err)
    results, metadata_list = check_correctness(
        sandbox_fusion_url=sandbox_url,
        in_outs=in_outs,
        generation=code,
        timeout=timeout_s,
        memory_limit_mb=1024,
        language="python",
        concurrent_semaphore=concurrent_semaphore,
    )
    
    passed = sum(1 for r in results if r is True)
    total = len(results)
    return passed, total, ""

def compute_score_deepcoder(sample_or_solution: dict, ground_truth: Any = None, **kwargs) -> Union[float, Dict[str, float], List[float]]:
    beta = float(kwargs.get("beta", 1.0))
    gamma = float(kwargs.get("gamma", 1.0))
    M_top = int(kwargs.get("M_top", 25))
    w1 = float(kwargs.get("w1", 0.7))
    w2 = float(kwargs.get("w2", 0.3))
    timeout_s = int(kwargs.get("timeout_s", 10))
    enable_thought = bool(kwargs.get("enable_thought", True))
    perf_gate = float(kwargs.get("perf_gate", 0.0))
    sandbox_url = kwargs.get("sandbox_url") or "http://localhost:8080/sandbox"  # allow explicit None to fall back
    concurrent_semaphore = kwargs.get("concurrent_semaphore", None)

    # Disable action trace for DeepCoder initially because the remote sandbox doesn't return AST trace sums
    # Would need custom sandbox modifications to support tracing
    enable_action = False 

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

    def _score_one(resp: str) -> Dict[str, float]:
        code = _extract_code(str(resp))
        passed, total, _ = _run_deepcoder_eval(code, inputs, expected_outputs, timeout_s, sandbox_url, concurrent_semaphore=concurrent_semaphore)
        S_perf = 0.0 if total == 0 else float(passed) / float(total)
        S_thought = 0.0
        S_action = 0.0
        final_reward = 0.0

        if S_perf > perf_gate:
            S_thought = compute_thought_score(code, M_top=M_top, w1=w1, w2=w2) if enable_thought else 0.0
            S_action = 0.0  # Disabled tracing inside remote sandbox for now
            final_reward = combine_reward(S_perf, S_thought, S_action, beta=beta, gamma=gamma)

        return {
            "score": float(final_reward),
            "combined_reward": float(final_reward),
            "acc": float(S_perf),
            "original_reward": float(S_perf),
            "thought_reward": float(S_thought),
            "action_reward": float(S_action),
        }

    if isinstance(sample.get("responses", None), list):
        return [_score_one(r)["score"] for r in sample["responses"]]

    for k in ("response", "completion", "output", "generated_text", "text"):
        if isinstance(sample.get(k, None), str) and sample[k].strip():
            return _score_one(sample[k])

    return 0.0
