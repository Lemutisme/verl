from typing import Any

from ..common import to_float
from .executable_verifier import cap_score_if_wrong, get_verification_report


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

    score = 0.50 * valid_ratio + 0.30 * unique_ratio + 0.20 * no_trailing_chatter
    return cap_score_if_wrong(ctx, score, kwargs)
