import ast
import re
from typing import Any


CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return default


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def extract_code(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    match = CODEBLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    unclosed = re.search(r"```(?:python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()
    return text.strip()


def parse_python(code: str) -> ast.AST | None:
    try:
        return ast.parse(code or "")
    except SyntaxError:
        return None


def response_text(ctx: dict[str, Any]) -> str:
    for key in ("response", "solution_str", "completion", "output", "generated_text", "text"):
        value = ctx.get(key)
        if isinstance(value, str):
            return value
    return ""
