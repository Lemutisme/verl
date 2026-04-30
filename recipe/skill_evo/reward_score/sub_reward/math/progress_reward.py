import re
from typing import Any

from ..common import response_text


def compute(ctx: dict[str, Any], **_: Any) -> float:
    text = response_text(ctx)
    if not text.strip():
        return 0.0
    sentences = [s for s in re.split(r"[\n.;。]+", text) if s.strip()]
    operators = len(re.findall(r"[+\-*/=]|\\frac|\\sqrt|\^", text))
    transition = len(re.findall(r"(?i)\b(therefore|thus|so|hence|then|because)\b", text))
    score = 0.20 + min(len(sentences), 8) * 0.04 + min(operators, 12) * 0.03 + min(transition, 5) * 0.04
    return score
