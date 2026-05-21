from typing import Any

from .executable_verifier import cap_score_if_wrong, get_verification_report


def compute(ctx: dict[str, Any], **kwargs: Any) -> float:
    report = get_verification_report(ctx, **kwargs)
    total = len(report.claims)
    score = report.valid_claim_count / float(total) if total > 0 else 0.0
    return cap_score_if_wrong(ctx, score, kwargs)
