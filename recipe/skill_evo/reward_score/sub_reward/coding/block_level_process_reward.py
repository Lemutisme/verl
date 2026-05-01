import ast
from typing import Any

from ..common import extract_code


def compute(ctx: dict[str, Any], **_: Any) -> float:
    code = ctx.get("code") or extract_code(ctx.get("response", ""))
    lines = [line for line in str(code).splitlines() if line.strip()]
    if not lines:
        return 0.0

    # Incremental prefix validity check — O(n) parse calls instead of O(n²)
    checkpoints = []
    prefix = ""
    for line in lines:
        prefix = prefix + line + "\n" if prefix else line + "\n"
        try:
            compile(prefix, "<string>", "exec")
            checkpoints.append(1.0)
        except SyntaxError:
            checkpoints.append(0.0)

    parse_prefix_score = sum(checkpoints) / float(len(checkpoints))
    final_bonus = 1.0 if checkpoints[-1] > 0 else 0.0
    return 0.7 * parse_prefix_score + 0.3 * final_bonus
