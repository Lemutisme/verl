import sys
from pathlib import Path
import base64
import json
import pickle
import zlib

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def test_pdar_deepcoder_acc_is_strict_and_keeps_partial_metrics(monkeypatch):
    import custom_reward

    def fake_deepcoder_score(*_, **__):
        return {
            "main_reward": 0.5,
            "subrewards": {"coding_compiler_runtime_feedback": 0.5},
        }

    monkeypatch.setattr(
        custom_reward.deepcoder_evaluator,
        "compute_score_deepcoder",
        fake_deepcoder_score,
    )

    info = custom_reward.compute_score(
        "deepcoder_unit",
        "solution",
        {"tests": "[]"},
        combine_mode="pdar",
        coding_enable_sub_rewards=True,
    )

    assert info["main_reward"] == pytest.approx(0.5)
    assert info["partial_pass_rate"] == pytest.approx(0.5)
    assert info["any_pass"] is True
    assert info["acc"] is False


def test_deepcoder_extracts_raw_code_after_explanation_without_fence():
    from reward_score import deepcoder_action_thought_reward as deepcoder

    response = (
        "We can solve it directly.\n\n"
        "import sys\n"
        "s = sys.stdin.read().strip()\n"
        "print(s[::-1])\n"
    )

    assert deepcoder._extract_code(response) == (
        "import sys\n"
        "s = sys.stdin.read().strip()\n"
        "print(s[::-1])"
    )


def test_deepcoder_normalizes_list_style_stdio_tests():
    from reward_score import deepcoder_action_thought_reward as deepcoder

    tests = json.dumps({"inputs": [["abc def"]], "outputs": [["cbafed"]]})

    parsed = deepcoder._get_tests_deepcoder({"tests": tests})

    assert parsed["inputs"] == ["abc def"]
    assert parsed["outputs"] == ["cbafed"]
    assert parsed["fn_name"] is None


def test_deepcoder_decodes_compressed_lcb_tests():
    from reward_score import deepcoder_action_thought_reward as deepcoder

    payload = json.dumps(
        {
            "inputs": ["1\nabc\n"],
            "outputs": ["YES\n"],
            "fn_name": None,
        }
    )
    encoded = base64.b64encode(zlib.compress(pickle.dumps(payload))).decode()

    parsed = deepcoder._get_tests_deepcoder({"tests": encoded})

    assert parsed["inputs"] == ["1\nabc\n"]
    assert parsed["outputs"] == ["YES\n"]


def test_compiler_runtime_feedback_does_not_reward_zero_passes():
    from reward_score.sub_reward.coding import compiler_runtime_feedback

    assert compiler_runtime_feedback.compute({"eval_total": 4, "eval_passed": 0}) == 0.0
    assert compiler_runtime_feedback.compute({"eval_total": 4, "eval_passed": 2}) == pytest.approx(0.5)
    assert compiler_runtime_feedback.compute({"eval_total": 4, "eval_passed": 4}) == 1.0


def test_executed_token_credit_without_coverage_uses_pass_rate_only():
    from reward_score.sub_reward.coding import executed_token_credit

    ctx = {
        "code": "def solve():\n    return 1\n",
        "eval_total": 4,
        "eval_passed": 1,
    }
    assert executed_token_credit.compute(ctx) == pytest.approx(0.25)


def test_static_and_block_coding_rewards_are_not_enabled_by_default():
    from reward_score.sub_reward import DEFAULT_ENABLED, DEFAULT_WEIGHTS

    assert DEFAULT_ENABLED["coding"]["static_analysis_reward"] is False
    assert DEFAULT_ENABLED["coding"]["block_level_process_reward"] is False
    assert DEFAULT_WEIGHTS["coding"]["static_analysis_reward"] == 0.0
    assert DEFAULT_WEIGHTS["coding"]["block_level_process_reward"] == 0.0


def test_math_executable_preset_uses_revised_live_rewards_by_default():
    script = (PROJECT_DIR / "run_grpo_math.sh").read_text()

    assert "MATH_ENABLE_FINAL_ANSWER_REWARD=${MATH_ENABLE_FINAL_ANSWER_REWARD:-false}" in script
    assert "MATH_ENABLE_ANSWER_EFFICIENCY_REWARD=${MATH_ENABLE_ANSWER_EFFICIENCY_REWARD:-false}" in script
    assert "MATH_ENABLE_CONSISTENCY_REWARD=${MATH_ENABLE_CONSISTENCY_REWARD:-false}" in script
    assert (
        "MATH_ENABLE_EXECUTABLE_UNIT_PASS_RATE_REWARD="
        "${MATH_ENABLE_EXECUTABLE_UNIT_PASS_RATE_REWARD:-false}"
    ) in script
    assert (
        "MATH_WEIGHT_EXECUTABLE_UNIT_PASS_RATE_REWARD="
        "${MATH_WEIGHT_EXECUTABLE_UNIT_PASS_RATE_REWARD:-0.0}"
    ) in script
    assert "MATH_WEIGHT_STEP_ARITHMETIC_VALIDITY_REWARD=${MATH_WEIGHT_STEP_ARITHMETIC_VALIDITY_REWARD:-0.35}" in script
    assert "MATH_WEIGHT_PREFIX_CONSISTENCY_REWARD=${MATH_WEIGHT_PREFIX_CONSISTENCY_REWARD:-0.25}" in script
    assert "MATH_WEIGHT_TRACE_EFFICIENCY_REWARD=${MATH_WEIGHT_TRACE_EFFICIENCY_REWARD:-0.25}" in script
    assert "MATH_WEIGHT_ANSWER_EXTRACTABILITY_REWARD=${MATH_WEIGHT_ANSWER_EXTRACTABILITY_REWARD:-0.15}" in script
    assert "PDAR_TAU_C_DEFAULT=0.30" in script


def test_math_pdar_ori_mode_uses_ori_reward_with_pdar_advantage():
    script = (PROJECT_DIR / "run_grpo_math.sh").read_text()
    mode_start = script.index("  pdar-ori|pdar_ori|ori-pdar|ori_pdar|pdar_original)")
    mode_end = script.index("  pdar|pdar_reward)", mode_start)
    mode_block = script[mode_start:mode_end]

    assert "pdar-ori" in script
    assert 'REWARD_LABEL="pdar-ori"' in mode_block
    assert 'COMBINE_MODE="none"' in mode_block
    assert 'ADV_ESTIMATOR="pdar"' in mode_block
    assert "MATH_ENABLE_SUB_REWARDS=${MATH_ENABLE_SUB_REWARDS:-false}" in mode_block


def test_deepcoder_pdar_script_defaults_to_non_saturated_aux_rewards():
    script = (PROJECT_DIR / "run_grpo.sh").read_text()
    pdar_start = script.index("  pdar|pdar_reward)")
    pdar_end = script.index("  *)", pdar_start)
    pdar_block = script[pdar_start:pdar_end]

    assert "DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-false}" in pdar_block
    assert "DEEPCODER_BETA=${DEEPCODER_BETA:-0.0}" in pdar_block
    assert "DEEPCODER_GAMMA=${DEEPCODER_GAMMA:-0.0}" in pdar_block
    assert "CODING_ENABLE_STATIC_ANALYSIS_REWARD=${CODING_ENABLE_STATIC_ANALYSIS_REWARD:-false}" in script
    assert "CODING_WEIGHT_STATIC_ANALYSIS_REWARD=${CODING_WEIGHT_STATIC_ANALYSIS_REWARD:-0.0}" in script
    assert "CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD=${CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD:-false}" in script
    assert "CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD=${CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD:-0.0}" in script


def test_deepcoder_script_prefers_cleaned_train_and_eval_data():
    script = (PROJECT_DIR / "run_grpo.sh").read_text()

    assert "deepcoder_full_train_clean.parquet" in script
    assert "code_eval_master_clean.parquet" in script
    assert "DEEPCODER_TRAIN_FILE" in script
    assert "DEEPCODER_VAL_FILE" in script


def test_deepcoder_pdar_ori_mode_uses_ori_reward_with_pdar_advantage():
    script = (PROJECT_DIR / "run_grpo.sh").read_text()
    mode_start = script.index("  pdar-ori|pdar_ori|ori-pdar|ori_pdar|pdar_original)")
    mode_end = script.index("  pdar|pdar_reward)", mode_start)
    mode_block = script[mode_start:mode_end]

    assert "pdar-ori" in script
    assert 'REWARD_LABEL="pdar-ori"' in mode_block
    assert 'COMBINE_MODE="none"' in mode_block
    assert 'ADV_ESTIMATOR="pdar"' in mode_block
    assert "CODING_ENABLE_SUB_REWARDS=${CODING_ENABLE_SUB_REWARDS:-false}" in mode_block
    assert "DEEPCODER_ENABLE_THOUGHT=${DEEPCODER_ENABLE_THOUGHT:-false}" in mode_block


def test_math_script_prefers_formatted_deepscalar_train_data():
    script = (PROJECT_DIR / "run_grpo_math.sh").read_text()

    assert "deepscalar_train_formatted.parquet" in script
    assert "DEEPSCALAR_TRAIN_FILE" in script


def test_math_script_prefers_formatted_general365_train_data():
    script = (PROJECT_DIR / "run_grpo_math.sh").read_text()

    assert "general365/train_formatted.parquet" in script
    assert "GENERAL365_TRAIN_FILE" in script


def test_math_prompt_formatter_can_normalize_boxed_general365_instruction_to_hash():
    from data_preprocess.format_math_prompts import format_math_rows

    rows = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Question\n"
                        "Output your final answer at the end of your reply using the following format:\n"
                        "### The final answer is: $\\boxed{<Your Answer>}$\n"
                        "For example:\n"
                        "### The final answer is: $\\boxed{123}$ "
                        "Let's think step by step and output the final answer within \\boxed{}."
                    ),
                }
            ],
        }
    ]

    formatted = format_math_rows(rows, force_hash=True)
    content = formatted[0]["prompt"][0]["content"]

    assert 'output the final answer after "####"' in content
    assert "\\boxed" not in content


def test_run_multiple_exp_accepts_pdar_ori_filter_without_expanding_default_matrix():
    script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    assert "[-reward {pdar|pd|new|ori|pdar-ori|pdpo}]" in script
    assert 'REWARDS=("pdar" "pd" "new" "ori")' in script
    assert 'pdar-ori|pdar_ori|ori-pdar|ori_pdar|pdar_original)' in script
    assert 'REWARDS=("pdar-ori")' in script


def test_pdpo_modes_are_available_but_not_added_to_default_matrix():
    math_script = (PROJECT_DIR / "run_grpo_math.sh").read_text()
    code_script = (PROJECT_DIR / "run_grpo.sh").read_text()
    multi_script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    assert "pdpo" in math_script
    assert 'REWARD_LABEL="pdpo"' in math_script
    assert 'ADV_ESTIMATOR="pdpo"' in math_script
    assert 'COMBINE_MODE="pdar"' in math_script

    assert "pdpo" in code_script
    assert 'REWARD_LABEL="pdpo"' in code_script
    assert 'ADV_ESTIMATOR="pdpo"' in code_script
    assert 'COMBINE_MODE="pdar"' in code_script

    assert "[-reward {pdar|pd|new|ori|pdar-ori|pdpo}]" in multi_script
    assert 'REWARDS=("pdar" "pd" "new" "ori")' in multi_script
    assert 'pdpo|pdpo_reward)' in multi_script
    assert 'REWARDS=("pdpo")' in multi_script
