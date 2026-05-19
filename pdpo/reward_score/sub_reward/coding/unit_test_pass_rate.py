from typing import Any


def compute(ctx: dict[str, Any], **_: Any) -> float:
    total = int(ctx.get("eval_total") or 0)
    passed = int(ctx.get("eval_passed") or 0)
    if total <= 0:
        return float(ctx.get("s_perf") or 0.0)
    return passed / float(total)
