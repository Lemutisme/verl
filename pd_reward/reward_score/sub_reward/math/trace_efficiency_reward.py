from typing import Any

from ..common import response_text, to_float
from .executable_verifier import cap_score_if_wrong, get_verification_report
from .efficiency_utils import approx_token_count, efficiency_bounds, length_score


def compute(ctx: dict[str, Any], **kwargs: Any) -> float:
    report = get_verification_report(ctx, **kwargs)
    total = len(report.claims)
    if total <= 0:
        return 0.0

    valid_ratio = report.valid_claim_count / float(total)
    unique_ratio = (
        report.unique_valid_claim_count / float(report.valid_claim_count)
        if report.valid_claim_count > 0
        else 0.0
    )
    post_answer_max = to_float(kwargs.get("math_efficiency_post_answer_max_tokens"), 24.0)
    if post_answer_max <= 0:
        no_trailing_chatter = 1.0 if report.post_answer_tokens == 0 else 0.0
    else:
        no_trailing_chatter = max(0.0, 1.0 - report.post_answer_tokens / post_answer_max)

    _, default_max_tokens, _ = efficiency_bounds(ctx, kwargs)
    soft_max_tokens = to_float(kwargs.get("math_trace_efficiency_soft_max_tokens"), default_max_tokens)
    hard_max_tokens = to_float(
        kwargs.get("math_trace_efficiency_hard_max_tokens"),
        max(soft_max_tokens + 1.0, 3.0 * soft_max_tokens),
    )
    overall_length = length_score(approx_token_count(response_text(ctx)), soft_max_tokens, hard_max_tokens)

    structural_score = 0.50 * valid_ratio + 0.30 * unique_ratio + 0.20 * no_trailing_chatter
    score = structural_score * (0.25 + 0.75 * overall_length)
    return cap_score_if_wrong(ctx, score, kwargs)
