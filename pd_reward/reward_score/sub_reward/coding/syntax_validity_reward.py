from typing import Any

from ..common import extract_code


def compute(ctx: dict[str, Any], **_: Any) -> float:
    if "code_compile_ok" in ctx:
        return 1.0 if ctx.get("code_compile_ok") else 0.0

    code = ctx.get("code") or extract_code(ctx.get("response", ""))
    if not str(code or "").strip():
        return 0.0
    try:
        compile(str(code), "<solution>", "exec")
    except Exception:
        return 0.0
    return 1.0
