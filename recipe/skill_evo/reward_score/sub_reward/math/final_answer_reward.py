from typing import Any


def compute(ctx: dict[str, Any], **_: Any) -> float:
    if "base_acc" in ctx:
        return 1.0 if bool(ctx["base_acc"]) else 0.0
    return 1.0 if float(ctx.get("base_score") or 0.0) > 0.0 else 0.0
