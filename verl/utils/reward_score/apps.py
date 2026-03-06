import base64
import json
import pickle
import zlib
from typing import Any


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _try_decode_compressed(payload: str) -> Any:
    decoded = base64.b64decode(payload.encode("utf-8"))
    decompressed = zlib.decompress(decoded)

    # Common pattern: pickle.dumps(json.dumps(test_cases))
    try:
        value = pickle.loads(decompressed)
    except Exception:
        value = decompressed

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    return value


def _normalize_test_cases(test_cases: Any) -> dict:
    value = test_cases

    if isinstance(value, dict):
        if "input_output" in value and ("inputs" not in value or "outputs" not in value):
            value = value["input_output"]
        elif "test_cases" in value and ("inputs" not in value or "outputs" not in value):
            value = value["test_cases"]

    if not isinstance(value, dict):
        if isinstance(value, str):
            try:
                value = _maybe_json_loads(value)
            except Exception:
                value = _try_decode_compressed(value)
        else:
            raise ValueError("Unsupported APPS test case format")

    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, dict):
        raise ValueError("APPS test cases must be a dict")

    if "assert_case" in value and isinstance(value.get("assert_case"), list):
        assert_cases = value["assert_case"]
        value.setdefault("inputs", ["" for _ in assert_cases])
        value.setdefault("outputs", [None for _ in assert_cases])

    if "inputs" not in value or "outputs" not in value:
        raise ValueError("APPS test cases missing `inputs` or `outputs`")

    inputs = value["inputs"]
    outputs = value["outputs"]
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("APPS test cases require list `inputs` and `outputs`")
    if len(inputs) != len(outputs):
        raise ValueError("APPS test cases length mismatch between `inputs` and `outputs`")

    return value


def compute_score(
    completion: str,
    ground_truth: Any,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore=None,
    memory_limit_mb: int | None = None,
    continuous: bool = True,
    timeout: int = 10,
):
    continuous = _to_bool(continuous, default=True)
    timeout = int(timeout)
    test_cases = _normalize_test_cases(ground_truth)

    if sandbox_fusion_url:
        from . import sandbox_fusion

        return sandbox_fusion.compute_score(
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
            completion=completion,
            test_cases=test_cases,
            continuous=continuous,
            timeout=timeout,
        )

    from . import prime_code

    return prime_code.compute_score(
        completion=completion,
        test_cases=test_cases,
        continuous=continuous,
    )

