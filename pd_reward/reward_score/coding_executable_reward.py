import base64
import json
import os
import pickle
import re
import subprocess
import tempfile
import zlib
from typing import Any, Dict, List, Tuple, Union

from verl.utils.reward_score.sandbox_fusion.utils import check_correctness

from reward_score.sub_reward import collect_subrewards


_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = _CODEBLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    unclosed = re.search(r"```(?:python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()

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


def parse_io_tests(sample: Dict[str, Any]) -> Dict[str, Any]:
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


def parse_assert_tests(sample: Dict[str, Any]) -> List[str]:
    for key in ("test_list", "tests", "test", "unit_tests"):
        value = sample.get(key)
        if value is None and isinstance(sample.get("meta"), dict):
            value = sample["meta"].get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            payload = _load_tests_payload(value)
            if isinstance(payload, list):
                return [str(item) for item in payload if str(item).strip()]
            if isinstance(payload, dict):
                for nested_key in ("test_list", "tests", "test", "unit_tests"):
                    nested = payload.get(nested_key)
                    if isinstance(nested, list):
                        return [str(item) for item in nested if str(item).strip()]
                    if isinstance(nested, str) and nested.strip():
                        return [line.strip() for line in nested.splitlines() if line.strip()]
            return [line.strip() for line in value.splitlines() if line.strip()]
    return []


def _compile_error(code: str) -> str:
    if not str(code or "").strip():
        return "empty_code"
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError:
        return "syntax_error"
    except Exception:
        return "compile_error"
    return ""


def _run_assert_eval(code: str, tests: List[str], timeout_s: int = 6) -> Tuple[int, int, str]:
    total = len(tests)
    if total == 0:
        return 0, 0, "no_tests"

    compile_error = _compile_error(code)
    if compile_error:
        return 0, total, compile_error

    runner = f"""
import sys
passed = 0
total = {total}
g = {{}}

code = {json.dumps(code)}
tests = {json.dumps(tests)}

try:
    exec(code, g)
except Exception:
    print("RESULT", 0, total, "runtime_error")
    sys.exit(0)

for t in tests:
    try:
        exec(t, g)
        passed += 1
    except AssertionError:
        pass
    except Exception:
        pass

print("RESULT", passed, total, "")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "coding_assert_eval.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(runner)

        try:
            proc = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            return 0, total, "timeout"

    match = re.search(r"RESULT\s+(\d+)\s+(\d+)\s+([^\n]*)", out)
    if not match:
        return 0, total, "no_result"
    return int(match.group(1)), int(match.group(2)), match.group(3).strip()


def _run_stdio_eval(
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

    compile_error = _compile_error(code)
    if compile_error:
        return 0, len(inputs), compile_error

    in_outs = {"inputs": inputs, "outputs": expected_outputs}
    if fn_name:
        in_outs["fn_name"] = fn_name
    if assert_cases is not None:
        in_outs["assert_case"] = assert_cases

    results, _ = check_correctness(
        sandbox_fusion_url=sandbox_url,
        in_outs=in_outs,
        generation=code,
        timeout=timeout_s,
        memory_limit_mb=1024,
        language="python",
        concurrent_semaphore=concurrent_semaphore,
    )

    passed = sum(1 for result in results if result is True)
    total = len(results)
    eval_error = "runtime_error" if any(result == -1 for result in results) else ""
    return passed, total, eval_error


def _sample_from_args(sample_or_solution: Union[Dict[str, Any], str], ground_truth: Any = None) -> Dict[str, Any]:
    if isinstance(sample_or_solution, dict) and ground_truth is None:
        return dict(sample_or_solution)

    sample: Dict[str, Any] = {}
    if isinstance(ground_truth, dict):
        sample.update(ground_truth)
    elif isinstance(ground_truth, list):
        sample["tests"] = ground_truth
    elif isinstance(ground_truth, str):
        sample["tests"] = ground_truth
    sample["response"] = str(sample_or_solution)
    return sample


def _score_one_stdio(resp: str, sample: Dict[str, Any], kwargs: dict[str, Any]) -> tuple[float, dict[str, Any], str]:
    tests = parse_io_tests(sample)
    sandbox_url = kwargs.get("sandbox_url") or "http://localhost:8080/sandbox"
    passed, total, eval_error = _run_stdio_eval(
        resp,
        tests["inputs"],
        tests["outputs"],
        int(kwargs.get("timeout_s", 10)),
        sandbox_url,
        concurrent_semaphore=kwargs.get("concurrent_semaphore"),
        fn_name=tests.get("fn_name"),
        assert_cases=tests.get("assert_case"),
    )
    return (0.0 if total == 0 else passed / float(total)), {"passed": passed, "total": total}, eval_error


def _score_one_assert(resp: str, sample: Dict[str, Any], kwargs: dict[str, Any]) -> tuple[float, dict[str, Any], str]:
    tests = parse_assert_tests(sample)
    passed, total, eval_error = _run_assert_eval(
        resp,
        tests,
        timeout_s=int(kwargs.get("timeout_s", 6)),
    )
    return (0.0 if total == 0 else passed / float(total)), {"passed": passed, "total": total}, eval_error


def compute_score_coding(
    sample_or_solution: Union[Dict[str, Any], str],
    ground_truth: Any = None,
    *,
    eval_mode: str = "stdio",
    **kwargs: Any,
) -> Union[float, Dict[str, float], List[float]]:
    sample = _sample_from_args(sample_or_solution, ground_truth)
    mode = str(eval_mode or "stdio").strip().lower()

    def _score_one(response: str) -> Dict[str, Any]:
        code = _extract_code(str(response))
        if mode in {"assert", "mbpp", "assert_tests"}:
            s_perf, eval_counts, eval_error = _score_one_assert(code, sample, kwargs)
        else:
            s_perf, eval_counts, eval_error = _score_one_stdio(code, sample, kwargs)

        compile_error = _compile_error(code)
        sub_ctx = {
            "response": response,
            "code": code,
            "sample": sample,
            "ground_truth": ground_truth,
            "s_perf": s_perf,
            "eval_passed": eval_counts["passed"],
            "eval_total": eval_counts["total"],
            "eval_error": eval_error,
            "code_present": bool(code.strip()),
            "code_compile_error": compile_error,
            "code_compile_ok": not compile_error,
        }
        subrewards = collect_subrewards("coding", sub_ctx, **kwargs)

        if kwargs.get("return_components", False):
            return {"main_reward": s_perf, "subrewards": subrewards}

        return {
            "score": float(s_perf),
            "combined_reward": float(s_perf),
            "acc": float(s_perf),
            "original_reward": float(s_perf),
            **{f"{name}_reward": float(value) for name, value in subrewards.items()},
        }

    if isinstance(sample.get("responses"), list):
        if kwargs.get("return_components", False):
            return [_score_one(response) for response in sample["responses"]]
        return [_score_one(response)["score"] for response in sample["responses"]]

    for key in ("response", "completion", "output", "generated_text", "text"):
        if isinstance(sample.get(key), str) and sample[key].strip():
            result = _score_one(sample[key])
            if kwargs.get("return_components", False):
                return result["main_reward"], result["subrewards"]
            return result

    empty_subrewards = collect_subrewards(
        "coding",
        {
            "response": "",
            "code": "",
            "sample": sample,
            "ground_truth": ground_truth,
            "s_perf": 0.0,
            "eval_passed": 0,
            "eval_total": 0,
            "eval_error": "empty_response",
            "code_present": False,
            "code_compile_error": "empty_code",
            "code_compile_ok": False,
        },
        **kwargs,
    )
    if kwargs.get("return_components", False):
        return 0.0, empty_subrewards
    return 0.0
