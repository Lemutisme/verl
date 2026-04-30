import re
from typing import Any

from ..common import response_text


BOX_RE = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def compute(ctx: dict[str, Any], **_: Any) -> float:
    text = response_text(ctx)
    if not text.strip():
        return 0.0
    if BOX_RE.search(text):
        return 1.0
    if re.search(r"(?i)(final answer|answer)\s*[:：]", text):
        return 0.55
    return 0.20
