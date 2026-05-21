from typing import Any


def compute(ctx: dict[str, Any], **_: Any) -> float:
    total = int(ctx.get("eval_total") or 0)
    passed = int(ctx.get("eval_passed") or 0)
    err = str(ctx.get("eval_error") or "").lower()
    if total > 0:
        return max(0.0, min(1.0, passed / float(total)))
    if "syntax" in err or "compile" in err or "parse" in err:
        return 0.05
    if "timeout" in err:
        return 0.0
    if "runtime" in err or "exception" in err or "error" in err:
        return 0.05
    return 0.0
