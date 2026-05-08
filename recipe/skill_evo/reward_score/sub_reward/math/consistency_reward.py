import re
from typing import Any

from .efficiency_utils import normalize_answer_text
from ..common import response_text


BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)")
NUMERIC_LINE_END_RE = re.compile(r"(?m)([-+]?\d+(?:\.\d+)?(?:/\d+)?)(?:\s*[.)\]]?)\s*$")


def _candidate_answers(text: str) -> list[str]:
    candidates = []
    for match in BOX_RE.finditer(text):
        candidates.append(match.group(1))
    for match in ANSWER_RE.finditer(text):
        candidates.append(match.group(1))
    for match in NUMERIC_LINE_END_RE.finditer(text):
        candidates.append(match.group(1))

    normalized = []
    for value in candidates:
        answer = normalize_answer_text(value)
        if answer and answer not in normalized:
            normalized.append(answer)
    return normalized


def compute(ctx: dict[str, Any], **_: Any) -> float:
    text = response_text(ctx)
    if not text.strip():
        return 0.0

    candidates = _candidate_answers(text)
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return 1.0

    final_answer = candidates[-1]
    aligned = sum(1 for value in candidates if value == final_answer)
    return aligned / float(len(candidates))
