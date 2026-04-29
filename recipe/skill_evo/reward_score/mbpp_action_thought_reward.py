# verl/verl/utils/reward_score/mbpp_action_thought_reward.py
import os
import re
import json
import tempfile
import subprocess
from typing import Any, Dict, List, Tuple, Union
import ast
import math
from collections import defaultdict


_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def _extract_code(text: str) -> str:
    """Prefer fenced code block; else return raw text."""
    if not isinstance(text, str):
        return ""
    m = _CODEBLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()

def _get_tests(sample: Dict[str, Any]) -> List[str]:
    # Common MBPP fields: test_list (list[str]) or tests (list[str]/str)
    for k in ("test_list", "tests", "test", "unit_tests"):
        v = sample.get(k, None)
        if v is None and isinstance(sample.get("meta", None), dict):
            v = sample["meta"].get(k, None)
        if v is None:
            continue
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            # sometimes it's a single string with newlines
            # split into lines that look like asserts
            lines = [ln.strip() for ln in v.splitlines() if ln.strip()]
            return lines
    return []


# =========================
# ADDED: Thought elegance (AST-only) helpers
# =========================
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


def compute_thought_score(
    code: str,
    M_top: int = 25,
    w1: float = 0.7,
    w2: float = 0.3,
) -> float:
    """Thought elegance in [0,1], computed only from AST. Parse fail -> 0."""
    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0

    depth_cnt, total, K = _ast_depth_stats(tree)

    # depth entropy -> concentration
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

    # top-level fan-out (module body length)
    top_body = getattr(tree, "body", None)
    F_top = len(top_body) if isinstance(top_body, list) else 0
    S_top = 1.0 - (min(F_top, M_top) / float(M_top))

    S = w1 * S_depth + w2 * S_top
    if S < 0.0:
        S = 0.0
    if S > 1.0:
        S = 1.0
    return float(S)


# =========================
# ADDED: Action elegance (trace-only) helpers
# =========================
def compute_action_score_from_sums(
    revisit_sum: float,
    cost_sum: float,
    nt: int,
    u1: float = 0.5,
    u2: float = 0.5,
    kappa: float = 8.0,
) -> float:
    """Action elegance in [0,1], computed from trace aggregates."""
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


# =========================
# ADDED: Correctness-gated reward combiner
# =========================
def combine_reward(
    S_perf: float,
    S_thought: float,
    S_action: float,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Reward in [0,1]. If S_perf=0 then reward=0."""
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


def _run_mbpp_tests_in_subproc(code: str, tests: List[str], timeout_s: int = 6) -> Tuple[int, int, str]:
    """
    Run code + tests in a fresh python process.
    Count passed asserts individually.
    """
    total = len(tests)
    if total == 0:
        return 0, 0, "no_tests"

    runner = f"""
import sys, traceback
passed = 0
total = {total}
g = {{}}

code = {json.dumps(code)}
tests = {json.dumps(tests)}

try:
    exec(code, g)
except Exception:
    # if code itself doesn't run, reward 0
    print("RESULT", 0, total)
    sys.exit(0)

for t in tests:
    try:
        exec(t, g)
        passed += 1
    except Exception:
        # ignore; treat as failed
        pass

print("RESULT", passed, total)
"""

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mbpp_eval.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(runner)

        try:
            p = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
            out = (p.stdout or "") + "\n" + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 0, total, "timeout"

    m = re.search(r"RESULT\s+(\d+)\s+(\d+)", out)
    if not m:
        return 0, total, "no_result"
    passed = int(m.group(1))
    total2 = int(m.group(2))
    return passed, total2, ""


# =========================
# ADDED: trace runner (action elegance stats) in a separate subprocess
# =========================
def _run_mbpp_tests_in_subproc_trace(
    code: str,
    tests: List[str],
    timeout_s: int = 8,
) -> Tuple[float, float, int, str]:
    """
    Run code + tests in a fresh python process, with tracing.
    Return: revisit_sum, cost_sum, nt, err_tag

    revisit per test = 1 - unique_lines / (steps + eps), clipped to [0,1]
    cost per test    = log(1 + steps)
    """
    nt = len(tests)
    if nt == 0:
        return 0.0, 0.0, 0, "no_tests"

    runner = f"""
import sys, math, json

revisit_sum = 0.0
cost_sum = 0.0
nt = {nt}
g = {{}}

code = {json.dumps(code)}
tests = {json.dumps(tests)}

def make_tracer(counter, uniq_lines):
    def _trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == "solution.py":
            counter[0] += 1
            uniq_lines.add(frame.f_lineno)
        return _trace
    return _trace

try:
    exec(compile(code, "solution.py", "exec"), g)
except Exception:
    print("TRACE_RESULT", 0.0, 0.0, nt)
    sys.exit(0)

for t in tests:
    counter = [0]
    uniq_lines = set()

    sys.settrace(make_tracer(counter, uniq_lines))
    try:
        exec(compile(t, "testcase.py", "exec"), g)
    except Exception:
        pass
    finally:
        sys.settrace(None)

    steps = counter[0]
    u = len(uniq_lines)

    if steps <= 0:
        revisit = 0.0
        cost = 0.0
    else:
        revisit = 1.0 - (u / (steps + 1e-6))
        if revisit < 0.0:
            revisit = 0.0
        if revisit > 1.0:
            revisit = 1.0
        cost = math.log(1.0 + steps)

    revisit_sum += revisit
    cost_sum += cost

print("TRACE_RESULT", revisit_sum, cost_sum, nt)
"""

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mbpp_trace_eval.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(runner)

        try:
            p = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
            out = (p.stdout or "") + "\n" + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 0.0, 0.0, nt, "timeout"

    m = re.search(
        r"TRACE_RESULT\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+(\d+)",
        out,
    )
    if not m:
        return 0.0, 0.0, nt, "no_result"

    revisit_sum = float(m.group(1))
    cost_sum = float(m.group(2))
    nt2 = int(m.group(3))
    return revisit_sum, cost_sum, nt2, ""


def compute_score_mbpp(
    sample_or_solution: Union[Dict[str, Any], str],
    ground_truth: Any = None,
    **kwargs,
) -> Union[float, List[float]]:
    """
    Flexible: supports either a sample dict or (solution_str, ground_truth).
    Returns reward(s) in [0,1].

    CHANGED: reward is correctness-gated combination:
      R = S_perf * (1 + beta*S_thought + gamma*S_action) / (1+beta+gamma)
    """

    # =========================
    # ADDED: knobs (safe defaults); does not change return types
    # =========================
    beta = float(kwargs.get("beta", 1.0))
    gamma = float(kwargs.get("gamma", 1.0))

    # thought hyperparams
    M_top = int(kwargs.get("M_top", 25))
    w1 = float(kwargs.get("w1", 0.7))
    w2 = float(kwargs.get("w2", 0.3))

    # action hyperparams
    u1 = float(kwargs.get("u1", 0.5))
    u2 = float(kwargs.get("u2", 0.5))
    kappa = float(kwargs.get("kappa", 8.0))

    # timeouts
    timeout_s = int(kwargs.get("timeout_s", 6))
    trace_timeout_s = int(kwargs.get("trace_timeout_s", 8))

    # toggles
    enable_action = bool(kwargs.get("enable_action", True))
    enable_thought = bool(kwargs.get("enable_thought", True))

    # optional: correctness gate to decide whether to run trace
    perf_gate = float(kwargs.get("perf_gate", 0.0))

    if isinstance(sample_or_solution, dict) and ground_truth is None:
        sample = sample_or_solution
    else:
        sample = {}
        if isinstance(ground_truth, dict):
            sample.update(ground_truth)
        elif isinstance(ground_truth, str):
            sample["tests"] = ground_truth
        sample["response"] = str(sample_or_solution)

    tests = _get_tests(sample)

    # =========================
    # ADDED: score helper (keeps original evaluation logic)
    # =========================
    def _score_one(resp: str) -> Dict[str, float]:
        code = _extract_code(str(resp))

        # correctness: ORIGINAL runner (unchanged)
        passed, total, _ = _run_mbpp_tests_in_subproc(code, tests, timeout_s=timeout_s)
        S_perf = 0.0 if total == 0 else float(passed) / float(total)
        S_thought = 0.0
        S_action = 0.0

        if S_perf > perf_gate:
            if enable_thought:
                S_thought = compute_thought_score(code, M_top=M_top, w1=w1, w2=w2)

            if enable_action:
                revisit_sum, cost_sum, nt, trace_err = _run_mbpp_tests_in_subproc_trace(
                    code, tests, timeout_s=trace_timeout_s
                )
                if not trace_err:
                    S_action = compute_action_score_from_sums(
                        revisit_sum=revisit_sum,
                        cost_sum=cost_sum,
                        nt=nt,
                        u1=u1,
                        u2=u2,
                        kappa=kappa,
                    )

        if kwargs.get("return_components", False):
            return {"main_reward": S_perf, "subrewards": {"thought": S_thought, "action": S_action}}

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
        }

    # Try common fields for generated text
    if isinstance(sample.get("responses", None), list):
        if kwargs.get("return_components", False):
            return [_score_one(r) for r in sample["responses"]]
        return [_score_one(r)["score"] for r in sample["responses"]]

    # Single response fallback
    for k in ("response", "completion", "output", "generated_text", "text"):
        if isinstance(sample.get(k, None), str) and sample[k].strip():
            res = _score_one(sample[k])
            if kwargs.get("return_components", False):
                return res["main_reward"], res["subrewards"]
            return res

    if kwargs.get("return_components", False):
        return 0.0, {"thought": 0.0, "action": 0.0}
    return 0.0
