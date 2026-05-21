import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from custom_reward import compute_score
from reward_score.sub_reward.math.executable_verifier import (
    UnsafeExpressionError,
    get_verification_report,
    safe_eval_fraction,
)
from reward_score.sub_reward.math import (
    executable_unit_pass_rate_reward,
    step_arithmetic_validity_reward,
)


def _ctx(response: str, *, answer: str, ground_truth: str = "72", base_acc: bool = True):
    return {
        "response": response,
        "solution_str": response,
        "ground_truth": ground_truth,
        "base_score": 1.0 if base_acc else 0.0,
        "base_acc": base_acc,
        "data_source": "openai/gsm8k",
        "extra_info": {
            "question": (
                "Natalia sold clips to 48 of her friends in April, and then she sold "
                "half as many clips in May. How many clips did Natalia sell altogether?"
            ),
            "answer": answer,
        },
    }


def test_safe_eval_accepts_numeric_arithmetic_forms():
    assert safe_eval_fraction("1,000 / 4") == 250
    assert safe_eval_fraction("-3 + 7/2") == pytest.approx(0.5)
    assert safe_eval_fraction("2^3 + 1") == 9


def test_safe_eval_rejects_unsupported_ast_and_division_by_zero():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_fraction("__import__('os').system('echo bad')")
    with pytest.raises(UnsafeExpressionError):
        safe_eval_fraction("x + 1")
    with pytest.raises(UnsafeExpressionError):
        safe_eval_fraction("1 / 0")


def test_report_extracts_gsm8k_gold_and_generated_equations():
    gold = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
        "#### 72"
    )
    response = (
        "She sold 48 / 2 = 24 in May.\n"
        "Then 48 + 24 = 72 total.\n"
        "#### 72"
    )

    report = get_verification_report(_ctx(response, answer=gold))

    assert len(report.gold_units) == 2
    assert len(report.non_final_gold_units) == 1
    assert report.matched_non_final_gold_units == 1
    assert report.valid_claim_count == 2
    assert report.prefix_validity_score == 1.0


def test_executable_unit_pass_excludes_final_answer_duplicate_but_keeps_intermediate():
    gold = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
        "#### 72"
    )

    only_final = _ctx("48 + 24 = 72\n#### 72", answer=gold)
    with_intermediate = _ctx("48 / 2 = 24\n48 + 24 = 72\n#### 72", answer=gold)

    assert executable_unit_pass_rate_reward.compute(only_final) == 0.0
    assert executable_unit_pass_rate_reward.compute(with_intermediate) == 1.0


def test_wrong_answer_caps_executable_subreward():
    gold = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
        "#### 72"
    )
    wrong_ctx = _ctx(
        "48 / 2 = 24\n48 + 24 = 72\n#### 73",
        answer=gold,
        ground_truth="72",
        base_acc=False,
    )

    assert step_arithmetic_validity_reward.compute(wrong_ctx, math_executable_wrong_cap=0.35) == 0.35


def test_custom_reward_pdar_smoke_returns_flattened_executable_metrics():
    gold = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
        "#### 72"
    )
    response = "48 / 2 = 24\n48 + 24 = 72\n#### 72"

    result = compute_score(
        "openai/gsm8k",
        response,
        "72",
        extra_info={"answer": gold, "question": "How many clips?"},
        combine_mode="pdar",
        math_enable_sub_rewards=True,
        math_enable_final_answer_reward=False,
        math_enable_answer_efficiency_reward=False,
        math_enable_consistency_reward=False,
        math_enable_executable_unit_pass_rate_reward=True,
        math_enable_step_arithmetic_validity_reward=True,
        math_enable_prefix_consistency_reward=True,
        math_enable_trace_efficiency_reward=True,
        math_enable_answer_extractability_reward=True,
        math_signed_reward=True,
    )

    assert "aux_reward_combined" in result
    assert "aux_rewards" in result
    assert "math_executable_unit_pass_rate_reward" in result
    assert "math_step_arithmetic_validity_reward" in result
    assert result["aux_rewards"]["math_executable_unit_pass_rate_reward"] == 1.0
