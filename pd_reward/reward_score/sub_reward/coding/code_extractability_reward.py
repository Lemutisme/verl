from typing import Any

from ..common import extract_code


def compute(ctx: dict[str, Any], **_: Any) -> float:
    if "code_present" in ctx:
        return 1.0 if ctx.get("code_present") else 0.0

    code = ctx.get("code") or extract_code(ctx.get("response", ""))
    return 1.0 if str(code or "").strip() else 0.0
