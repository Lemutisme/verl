import re
from typing import Any

from ..common import response_text


def compute(ctx: dict[str, Any], **_: Any) -> float:
    text = response_text(ctx)
    if not text.strip():
        return 0.0
    has_final = "\\boxed{" in text or re.search(r"(?i)(final answer|answer)\s*[:：]", text)
    derivation_units = len(re.findall(r"=|\\frac|\\sqrt|\^|\btherefore\b|\bthus\b|\bhence\b", text, flags=re.I))
    distance_proxy = 1.0 / (1.0 + max(0, 6 - derivation_units))
    if has_final:
        distance_proxy += 0.35
    return distance_proxy
