import math
import re
from typing import Any

from ..common import clip, response_text, to_float


ANSWER_RE = re.compile(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)")


def _is_correct(ctx: dict[str, Any]) -> bool:
    if "base_acc" in ctx:
        return bool(ctx["base_acc"])
    return float(ctx.get("base_score") or 0.0) > 0.0


def _approx_token_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    word_like = len(re.findall(r"\S+", text))
    char_like = int(math.ceil(len(re.sub(r"\s+", "", text)) / 4.0))
    return max(1, word_like, char_like)


def _last_boxed_span(text: str) -> tuple[int, int] | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    pos = start + len("\\boxed{")
    depth = 1
    while pos < len(text):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, pos + 1
        pos += 1
    return None


def _normalize_answer_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9.+\-*/=]+", "", text)
    return text


def _answer_end_index(text: str, ground_truth: Any) -> int:
    boxed = _last_boxed_span(text)
    if boxed is not None:
        return boxed[1]

    matches = list(ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].end(1)

    target = _normalize_answer_text(ground_truth)
    if target:
        compact_chars = []
        raw_indices = []
        for idx, ch in enumerate(text):
            if re.match(r"[a-zA-Z0-9.+\-*/=]", ch):
                compact_chars.append(ch.lower())
                raw_indices.append(idx)
        compact = "".join(compact_chars)
        pos = compact.rfind(target)
        if pos >= 0:
            return raw_indices[min(len(raw_indices) - 1, pos + len(target) - 1)] + 1

    return len(text)


def _length_score(tokens: int, min_tokens: float, max_tokens: float) -> float:
    if max_tokens <= min_tokens:
        return 1.0 if tokens <= min_tokens else 0.0
    return 1.0 - clip((tokens - min_tokens) / (max_tokens - min_tokens), 0.0, 1.0)


def _dataset_default_max_tokens(data_source: str) -> float:
    ds = data_source.lower()
    if "general365" in ds:
        return 320.0
    if "deepscalar" in ds:
        return 512.0
    return 384.0


def compute(ctx: dict[str, Any], **kwargs: Any) -> float:
    if not _is_correct(ctx):
        return 0.0

    text = response_text(ctx)
    if not text.strip():
        return 0.0

    data_source = str(ctx.get("data_source") or "")
    min_tokens = to_float(kwargs.get("math_efficiency_min_tokens"), 16.0)
    max_tokens = to_float(
        kwargs.get("math_efficiency_max_tokens"),
        _dataset_default_max_tokens(data_source),
    )
    post_answer_max_tokens = to_float(kwargs.get("math_efficiency_post_answer_max_tokens"), 24.0)

    answer_end = _answer_end_index(text, ctx.get("ground_truth"))
    answer_prefix = text[:answer_end]
    answer_suffix = text[answer_end:]

    tokens_to_answer = _approx_token_count(answer_prefix)
    total_tokens = _approx_token_count(text)
    post_answer_tokens = _approx_token_count(answer_suffix)

    answer_efficiency = _length_score(tokens_to_answer, min_tokens, max_tokens)
    total_efficiency = _length_score(total_tokens, min_tokens, max_tokens * 1.20)
    no_trailing_chatter = _length_score(post_answer_tokens, 0.0, post_answer_max_tokens)

    return (
        0.70 * answer_efficiency
        + 0.20 * total_efficiency
        + 0.10 * no_trailing_chatter
    )
