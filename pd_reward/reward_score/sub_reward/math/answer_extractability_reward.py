from typing import Any

from .executable_verifier import cap_score_if_wrong, get_verification_report


def compute(ctx: dict[str, Any], **kwargs: Any) -> float:
    report = get_verification_report(ctx, **kwargs)
    score = 1.0 if report.final_answer_extractable else 0.0
    return cap_score_if_wrong(ctx, score, kwargs)
