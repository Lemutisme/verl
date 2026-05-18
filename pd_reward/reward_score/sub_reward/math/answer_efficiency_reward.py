from typing import Any

from .efficiency_utils import approx_token_count, efficiency_bounds, length_score, response_parts


def compute(ctx: dict[str, Any], **kwargs: Any) -> float:
    text, answer_prefix, answer_suffix = response_parts(ctx)
    if not text.strip():
        return 0.0

    min_tokens, max_tokens, post_answer_max_tokens = efficiency_bounds(ctx, kwargs)

    tokens_to_answer = approx_token_count(answer_prefix)
    total_tokens = approx_token_count(text)
    post_answer_tokens = approx_token_count(answer_suffix)

    answer_efficiency = length_score(tokens_to_answer, min_tokens, max_tokens)
    total_efficiency = length_score(total_tokens, min_tokens, max_tokens * 1.20)
    if post_answer_max_tokens <= 0:
        no_trailing_chatter = 1.0 if post_answer_tokens == 0 else 0.0
    else:
        no_trailing_chatter = max(0.0, 1.0 - post_answer_tokens / post_answer_max_tokens)

    return 0.70 * answer_efficiency + 0.20 * total_efficiency + 0.10 * no_trailing_chatter
