from typing import Any


def compute(ctx: dict[str, Any], **_: Any) -> float:
    total = int(ctx.get("eval_total") or 0)
    passed = int(ctx.get("eval_passed") or 0)
    err = str(ctx.get("eval_error") or "").lower()
    if total > 0:
        ratio = passed / float(total)
        if passed == total:
            return 1.0
        return max(0.35, 0.35 + 0.65 * ratio)
    if "syntax" in err or "compile" in err or "parse" in err:
        return 0.10
    if "timeout" in err:
        return 0.20
    if "runtime" in err or "exception" in err or "error" in err:
        return 0.25
    return 0.30
