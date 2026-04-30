import re
from typing import Any

from ..common import response_text


STEP_MARKERS = re.compile(r"(?im)^\s*(?:step\s*\d+|\d+[\).]|[-*]\s+)")


def compute(ctx: dict[str, Any], **_: Any) -> float:
    text = response_text(ctx)
    markers = len(STEP_MARKERS.findall(text))
    equations = len(re.findall(r"[=<>]=?|\\frac|\\sqrt|\^", text))
    if not text.strip():
        return 0.0
    score = 0.25 + min(markers, 5) * 0.10 + min(equations, 8) * 0.04
    if "\\boxed{" in text or re.search(r"(?i)answer\s*[:：]", text):
        score += 0.15
    return score
