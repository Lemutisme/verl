from typing import Any


def compute(ctx: dict[str, Any], **_: Any) -> float:
    code = str(ctx.get("code") or "")
    total_lines = [line for line in code.splitlines() if line.strip()]
    executed = ctx.get("executed_lines")
    if isinstance(executed, (set, list, tuple)) and total_lines:
        return len(set(executed)) / float(len(total_lines))

    total = int(ctx.get("eval_total") or 0)
    if total <= 0:
        return 0.0

    return 0.0
