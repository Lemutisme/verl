import re
from typing import Any

from ..common import response_text


BAD_MARKERS = re.compile(r"(?i)\b(i guess|not sure|maybe|cannot solve|no idea)\b")


def compute(ctx: dict[str, Any], **_: Any) -> float:
    text = response_text(ctx)
    if not text.strip():
        return 0.0
    score = 0.55
    if "\\boxed{" in text:
        score += 0.15
    if re.search(r"(?i)\b(check|verify|substitute|simplify)\b", text):
        score += 0.10
    if len(re.findall(r"=", text)) >= 2:
        score += 0.10
    if BAD_MARKERS.search(text):
        score -= 0.25
    return score
