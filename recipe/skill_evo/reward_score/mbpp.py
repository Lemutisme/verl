# verl/verl/utils/reward_score/mbpp.py
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

def compute_score_mbpp(
    sample_or_solution: Union[Dict[str, Any], str],
    ground_truth: Any = None,
    **kwargs,
) -> Union[float, List[float]]:
    """
    Flexible: supports either a sample dict or (solution_str, ground_truth).
    Returns reward(s) in [0,1].
    """
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

    # Try common fields for generated text
    # VERL里常见："responses" 是 list[str]
    if isinstance(sample.get("responses", None), list):
        rewards: List[float] = []
        for r in sample["responses"]:
            code = _extract_code(str(r))
            passed, total, _ = _run_mbpp_tests_in_subproc(code, tests)
            rewards.append(0.0 if total == 0 else float(passed) / float(total))
        return rewards

    # Single response fallback
    for k in ("response", "completion", "output", "generated_text", "text"):
        if isinstance(sample.get(k, None), str) and sample[k].strip():
            code = _extract_code(sample[k])
            passed, total, _ = _run_mbpp_tests_in_subproc(code, tests)
            return 0.0 if total == 0 else float(passed) / float(total)

    return 0.0
