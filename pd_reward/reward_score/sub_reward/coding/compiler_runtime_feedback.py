from typing import Any


def compute(ctx: dict[str, Any], **_: Any) -> float:
    total = int(ctx.get("eval_total") or 0)
    passed = int(ctx.get("eval_passed") or 0)
    err = str(ctx.get("eval_error") or "").lower()
    if "syntax" in err or "compile" in err or "parse" in err:
        return 0.0
    if "timeout" in err:
        return 0.10
    if "runtime" in err or "exception" in err or "error" in err:
        return 0.20

    compile_ok = bool(ctx.get("code_compile_ok", False))
    if total <= 0:
        return 0.25 if compile_ok else 0.0

    pass_rate = max(0.0, min(1.0, passed / float(total)))
    if pass_rate <= 0.0:
        return 0.35 if compile_ok else 0.0
    return 0.50 + 0.50 * pass_rate
