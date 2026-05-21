import os
import re
import json
import base64
import pickle
import threading
import zlib
from typing import Any, Dict, List, Tuple, Union
import ast
import math
from collections import defaultdict

# Reuse the sandbox execution wrapper from sandbox_fusion
# (Assuming sandbox_fusion/utils.py is available in verl.utils.reward_score)
from verl.utils.reward_score.sandbox_fusion.utils import check_correctness, DEFAULT_TIMEOUT
from reward_score.sub_reward import collect_subrewards

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
    stripped = text.strip()
    lines = stripped.splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"\s*(import|from|def|class|if\s+__name__\s*==|[a-zA-Z_][\w_]*\s*=)", line):
            return "\n".join(lines[idx:]).strip()
    return stripped


def _load_tests_payload(tests_payload: Any) -> Any:
    if isinstance(tests_payload, (dict, list)):
        return tests_payload
    if not isinstance(tests_payload, str):
        return {}

    text = tests_payload.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        decoded = zlib.decompress(base64.b64decode(text))
        payload = pickle.loads(decoded)
        if isinstance(payload, bytes):
            payload = payload.decode()
        if isinstance(payload, str):
            return json.loads(payload)
        return payload
    except Exception:
        return {}


def _stringify_stdio_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _stringify_stdio_value(value[0])
        return "\n".join(_stringify_stdio_value(item).rstrip("\n") for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _stringify_function_input(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in value)
    return json.dumps(value, ensure_ascii=False)


def _stringify_function_output(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)) or value is None or isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _get_tests_deepcoder(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract inputs and outputs pairs from DeepCoder tests JSON string.
    """
    tests_str = sample.get("tests", "{}")
    if not tests_str:
        return {"inputs": [], "outputs": [], "fn_name": None, "assert_case": []}

    data = _load_tests_payload(tests_str)
    if isinstance(data, list):
        inputs = []
        outputs = []
        for item in data:
            if isinstance(item, dict):
                inputs.append(_stringify_stdio_value(item.get("input", item.get("stdin", ""))))
                outputs.append(_stringify_stdio_value(item.get("output", item.get("stdout", ""))))
        return {"inputs": inputs, "outputs": outputs, "fn_name": None, "assert_case": [""] * len(inputs)}

    if not isinstance(data, dict):
        return {"inputs": [], "outputs": [], "fn_name": None, "assert_case": []}

    raw_inputs = data.get("inputs", [])
    raw_outputs = data.get("outputs", [])
    fn_name = data.get("fn_name") or None

    if fn_name:
        inputs = [_stringify_function_input(value) for value in raw_inputs]
        outputs = [_stringify_function_output(value) for value in raw_outputs]
    else:
        inputs = [_stringify_stdio_value(value) for value in raw_inputs]
        outputs = [_stringify_stdio_value(value) for value in raw_outputs]

    assert_cases = data.get("assert_case") or [""] * len(inputs)
    assert_cases = [_stringify_stdio_value(value) for value in assert_cases]
    return {"inputs": inputs, "outputs": outputs, "fn_name": fn_name, "assert_case": assert_cases}

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

def _run_deepcoder_eval(
    code: str,
    inputs: List[str],
    expected_outputs: List[str],
    timeout_s: int = 10,
    sandbox_url: str = "http://localhost:8000/run",
    concurrent_semaphore=None,
    fn_name: str | None = None,
    assert_cases: List[str] | None = None,
) -> Tuple[int, int, str]:
    if not inputs:
        return 0, 0, "no_tests"

    in_outs = {"inputs": inputs, "outputs": expected_outputs}
    if fn_name:
        in_outs["fn_name"] = fn_name
    if assert_cases is not None:
        in_outs["assert_case"] = assert_cases
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
    
    tests = _get_tests_deepcoder(sample)
    inputs = tests["inputs"]
    expected_outputs = tests["outputs"]
    fn_name = tests.get("fn_name")
    assert_cases = tests.get("assert_case")

    def _score_one(resp: str) -> Dict[str, float]:
        code = _extract_code(str(resp))
        passed, total, eval_error = _run_deepcoder_eval(
            code,
            inputs,
            expected_outputs,
            timeout_s,
            sandbox_url,
            concurrent_semaphore=concurrent_semaphore,
            fn_name=fn_name,
            assert_cases=assert_cases,
        )
        S_perf = 0.0 if total == 0 else float(passed) / float(total)
        S_thought = 0.0
        S_action = 0.0

        if S_perf > perf_gate:
            S_thought = compute_thought_score(code, M_top=M_top, w1=w1, w2=w2) if enable_thought else 0.0

        subrewards = {"thought": S_thought, "action": S_action}
        sub_ctx = {
            "response": resp,
            "code": code,
            "sample": sample,
            "ground_truth": ground_truth,
            "s_perf": S_perf,
            "eval_passed": passed,
            "eval_total": total,
            "eval_error": eval_error,
        }
        subrewards.update(collect_subrewards("coding", sub_ctx, **kwargs))

        if kwargs.get("return_components", False):
            return {"main_reward": S_perf, "subrewards": subrewards}

        final_reward = 0.0
        if S_perf > perf_gate:
            final_reward = combine_reward(S_perf, S_thought, S_action, beta=beta, gamma=gamma)

        return {
            "score": float(final_reward),
            "combined_reward": float(final_reward),
            "acc": float(S_perf),
            "original_reward": float(S_perf),
            "thought_reward": float(S_thought),
            "action_reward": float(S_action),
            **{f"{name}_reward": float(value) for name, value in subrewards.items()},
        }

    if isinstance(sample.get("responses", None), list):
        if kwargs.get("return_components", False):
            return [_score_one(r) for r in sample["responses"]]
        return [_score_one(r)["score"] for r in sample["responses"]]

    for k in ("response", "completion", "output", "generated_text", "text"):
        if isinstance(sample.get(k, None), str) and sample[k].strip():
            res = _score_one(sample[k])
            if kwargs.get("return_components", False):
                return res["main_reward"], res["subrewards"]
            return res

    if kwargs.get("return_components", False):
        return 0.0, {"thought": 0.0, "action": 0.0}
    return 0.0
