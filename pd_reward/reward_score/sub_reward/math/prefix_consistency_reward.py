from typing import Any

from .executable_verifier import cap_score_if_wrong, get_verification_report


def compute(ctx: dict[str, Any], **kwargs: Any) -> float:
    report = get_verification_report(ctx, **kwargs)
    return cap_score_if_wrong(ctx, report.prefix_validity_score, kwargs)
