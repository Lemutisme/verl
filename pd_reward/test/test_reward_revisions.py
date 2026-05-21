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


def test_pdpo_deepcoder_acc_is_strict_and_keeps_partial_metrics(monkeypatch):
    import custom_reward

    def fake_coding_score(*_, **__):
        return {
            "main_reward": 0.5,
            "subrewards": {"coding_compiler_runtime_feedback": 0.5},
        }

    monkeypatch.setattr(
        custom_reward.coding_evaluator,
        "compute_score_coding",
        fake_coding_score,
    )

    info = custom_reward.compute_score(
        "deepcoder_unit",
        "solution",
        {"tests": "[]"},
        combine_mode="pdpo",
        coding_enable_sub_rewards=True,
    )

    assert info["main_reward"] == pytest.approx(0.5)
    assert info["partial_pass_rate"] == pytest.approx(0.5)
    assert info["any_pass"] is True
    assert info["acc"] is False


def test_deepcoder_extracts_raw_code_after_explanation_without_fence():
    from reward_score import coding_executable_reward as coding

    response = (
        "We can solve it directly.\n\n"
        "import sys\n"
        "s = sys.stdin.read().strip()\n"
        "print(s[::-1])\n"
    )

    assert coding._extract_code(response) == (
        "import sys\n"
        "s = sys.stdin.read().strip()\n"
        "print(s[::-1])"
    )


def test_deepcoder_normalizes_list_style_stdio_tests():
    from reward_score import coding_executable_reward as coding

    tests = json.dumps({"inputs": [["abc def"]], "outputs": [["cbafed"]]})

    parsed = coding.parse_io_tests({"tests": tests})

    assert parsed["inputs"] == ["abc def"]
    assert parsed["outputs"] == ["cbafed"]
    assert parsed["fn_name"] is None


def test_deepcoder_decodes_compressed_lcb_tests():
    from reward_score import coding_executable_reward as coding

    payload = json.dumps(
        {
            "inputs": ["1\nabc\n"],
            "outputs": ["YES\n"],
            "fn_name": None,
        }
    )
    encoded = base64.b64encode(zlib.compress(pickle.dumps(payload))).decode()

    parsed = coding.parse_io_tests({"tests": encoded})

    assert parsed["inputs"] == ["1\nabc\n"]
    assert parsed["outputs"] == ["YES\n"]


def test_compiler_runtime_feedback_does_not_reward_zero_passes():
    from reward_score.sub_reward.coding import compiler_runtime_feedback

    assert compiler_runtime_feedback.compute({"eval_total": 4, "eval_passed": 0}) == 0.0
    assert compiler_runtime_feedback.compute(
        {"eval_total": 4, "eval_passed": 0, "code_compile_ok": True}
    ) == pytest.approx(0.35)
    assert compiler_runtime_feedback.compute({"eval_total": 4, "eval_passed": 2}) == pytest.approx(0.75)
    assert compiler_runtime_feedback.compute({"eval_total": 4, "eval_passed": 4}) == 1.0


def test_executed_token_credit_without_coverage_does_not_duplicate_pass_rate():
    from reward_score.sub_reward.coding import executed_token_credit

    ctx = {
        "code": "def solve():\n    return 1\n",
        "eval_total": 4,
        "eval_passed": 1,
    }
    assert executed_token_credit.compute(ctx) == 0.0


def test_general_coding_components_do_not_emit_dataset_specific_thought_action(monkeypatch):
    from reward_score import coding_executable_reward as coding

    monkeypatch.setattr(coding, "_run_stdio_eval", lambda *_, **__: (0, 3, ""))

    main_reward, subrewards = coding.compute_score_coding(
        "print(1)",
        {"tests": json.dumps({"inputs": ["1", "2", "3"], "outputs": ["2", "3", "4"]})},
        return_components=True,
        coding_enable_sub_rewards=True,
    )

    assert main_reward == 0.0
    assert "thought" not in subrewards
    assert "action" not in subrewards
    assert subrewards["coding_code_extractability_reward"] == 1.0
    assert subrewards["coding_syntax_validity_reward"] == 1.0
    assert subrewards["coding_compiler_runtime_feedback"] > 0.0


def test_mbpp_and_eurus_sources_route_to_general_coding_evaluator(monkeypatch):
    import custom_reward

    calls = []

    def fake_general_coding_score(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "main_reward": 1.0,
            "subrewards": {"coding_syntax_validity_reward": 1.0},
        }

    monkeypatch.setattr(
        custom_reward.coding_evaluator,
        "compute_score_coding",
        fake_general_coding_score,
    )

    for source in ("mbpp", "code_contests"):
        info = custom_reward.compute_score(
            source,
            "print(1)",
            {"tests": json.dumps({"inputs": ["1"], "outputs": ["1"]})},
            combine_mode="pdpo",
            coding_enable_sub_rewards=True,
        )
        assert info["main_reward"] == 1.0
        assert info["acc"] is True

    assert [call[1]["eval_mode"] for call in calls] == ["assert", "stdio"]


def test_eurus_preprocess_uses_general_coding_test_parser():
    script = (PROJECT_DIR / "data_preprocess" / "prepare_eurus_data.py").read_text()

    assert "from reward_score.coding_executable_reward import parse_io_tests" in script
    assert "_get_tests_deepcoder" not in script


def test_static_and_block_coding_rewards_are_not_enabled_by_default():
    from reward_score.sub_reward import DEFAULT_ENABLED, DEFAULT_WEIGHTS

    assert DEFAULT_ENABLED["coding"]["code_extractability_reward"] is True
    assert DEFAULT_ENABLED["coding"]["syntax_validity_reward"] is True
    assert DEFAULT_ENABLED["coding"]["static_analysis_reward"] is False
    assert DEFAULT_ENABLED["coding"]["executed_token_credit"] is False
    assert DEFAULT_ENABLED["coding"]["block_level_process_reward"] is False
    assert DEFAULT_WEIGHTS["coding"]["static_analysis_reward"] == 0.0
    assert DEFAULT_WEIGHTS["coding"]["executed_token_credit"] == 0.0
    assert DEFAULT_WEIGHTS["coding"]["block_level_process_reward"] == 0.0


def test_math_executable_preset_uses_revised_live_rewards_by_default():
    script = (PROJECT_DIR / "train_math.sh").read_text()

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
    assert "MATH_WEIGHT_PREFIX_CONSISTENCY_REWARD=${MATH_WEIGHT_PREFIX_CONSISTENCY_REWARD:-0.15}" in script
    assert "MATH_WEIGHT_TRACE_EFFICIENCY_REWARD=${MATH_WEIGHT_TRACE_EFFICIENCY_REWARD:-0.35}" in script
    assert "MATH_WEIGHT_ANSWER_EXTRACTABILITY_REWARD=${MATH_WEIGHT_ANSWER_EXTRACTABILITY_REWARD:-0.15}" in script
    assert "PDPO_BETA_SAME=${PDPO_BETA_SAME:-0.70}" in script
    assert "PDPO_LAMBDA_AUX=${PDPO_LAMBDA_AUX:-0.70}" in script
    assert "PDPO_ANSWER_GATE_MIN=${PDPO_ANSWER_GATE_MIN:-0.5}" in script
    assert "pdpo_answer_gate_closed_scale" in script
    assert "PDPO_CORRECTNESS_SAFE=${PDPO_CORRECTNESS_SAFE:-true}" in script
    assert "PDPO_RELIABILITY_ENABLED=${PDPO_RELIABILITY_ENABLED:-true}" in script
    assert "pdpo_reliability_wrong_high_threshold" in script


def test_math_reward_presets_only_include_active_matrix():
    script = (PROJECT_DIR / "train_math.sh").read_text()

    assert "-reward {ori|new|pdpo|gdpo}" in script
    assert 'REWARD_KIND=${REWARD_KIND:-"pdpo"}' in script
    assert "pdar-ori|" not in script
    assert "  pd|primal_dual|pd_reward)" not in script
    assert "  pdar|pdar_reward)" not in script


def test_coding_pdpo_script_defaults_to_general_aux_rewards():
    script = (PROJECT_DIR / "train_code.sh").read_text()
    pdpo_start = script.index("  pdpo|pdpo_reward)")
    pdpo_end = script.index("  *)", pdpo_start)
    pdpo_block = script[pdpo_start:pdpo_end]

    assert "DEEPCODER_ENABLE_THOUGHT" not in script
    assert "DEEPCODER_BETA" not in script
    assert "DEEPCODER_GAMMA" not in script
    assert "CODING_PERF_GATE=${CODING_PERF_GATE:--1.0}" in script
    assert "CODING_ENABLE_CODE_EXTRACTABILITY_REWARD=${CODING_ENABLE_CODE_EXTRACTABILITY_REWARD:-true}" in script
    assert "CODING_WEIGHT_CODE_EXTRACTABILITY_REWARD=${CODING_WEIGHT_CODE_EXTRACTABILITY_REWARD:-0.15}" in script
    assert "CODING_ENABLE_SYNTAX_VALIDITY_REWARD=${CODING_ENABLE_SYNTAX_VALIDITY_REWARD:-true}" in script
    assert "CODING_WEIGHT_SYNTAX_VALIDITY_REWARD=${CODING_WEIGHT_SYNTAX_VALIDITY_REWARD:-0.25}" in script
    assert "CODING_ENABLE_SUB_REWARDS=${CODING_ENABLE_SUB_REWARDS:-true}" in pdpo_block
    assert 'ADV_ESTIMATOR="pdpo"' in pdpo_block
    assert 'COMBINE_MODE="pdpo"' in pdpo_block
    assert "CODING_ENABLE_STATIC_ANALYSIS_REWARD=${CODING_ENABLE_STATIC_ANALYSIS_REWARD:-false}" in script
    assert "CODING_WEIGHT_STATIC_ANALYSIS_REWARD=${CODING_WEIGHT_STATIC_ANALYSIS_REWARD:-0.0}" in script
    assert "CODING_ENABLE_EXECUTED_TOKEN_CREDIT=${CODING_ENABLE_EXECUTED_TOKEN_CREDIT:-false}" in script
    assert "CODING_WEIGHT_EXECUTED_TOKEN_CREDIT=${CODING_WEIGHT_EXECUTED_TOKEN_CREDIT:-0.0}" in script
    assert "CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD=${CODING_ENABLE_BLOCK_LEVEL_PROCESS_REWARD:-false}" in script
    assert "CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD=${CODING_WEIGHT_BLOCK_LEVEL_PROCESS_REWARD:-0.0}" in script


def test_coding_script_prefers_eurus_train_and_eval_data():
    script = (PROJECT_DIR / "train_code.sh").read_text()

    assert "eurus_code_train.parquet" in script
    assert "eurus_code_val.parquet" in script
    assert "EURUS_TRAIN_FILE" in script
    assert "EURUS_VAL_FILE" in script
    assert 'PROJECT_NAME=${PROJECT_NAME:-"eurus_grpo"}' in script


def test_eurus_coding_sources_route_to_executable_reward(monkeypatch):
    import custom_reward

    def fake_coding_score(*_, **__):
        return {
            "main_reward": 1.0,
            "subrewards": {"coding_compiler_runtime_feedback": 1.0},
        }

    monkeypatch.setattr(
        custom_reward.coding_evaluator,
        "compute_score_coding",
        fake_coding_score,
    )

    info = custom_reward.compute_score(
        "code_contests",
        "print(input())",
        '{"inputs": ["1\\n"], "outputs": ["1\\n"]}',
        combine_mode="pdpo",
        coding_enable_sub_rewards=True,
    )

    assert info["main_reward"] == 1.0
    assert info["acc"] is True


def test_coding_reward_presets_only_include_active_matrix():
    script = (PROJECT_DIR / "train_code.sh").read_text()

    assert "-reward {ori|new|pdpo|gdpo}" in script
    assert 'REWARD_KIND=${REWARD_KIND:-"pdpo"}' in script
    assert "pdar-ori|" not in script
    assert "  pd|primal_dual|pd_reward)" not in script
    assert "  pdar|pdar_reward)" not in script


def test_math_script_prefers_formatted_deepscalar_train_data():
    script = (PROJECT_DIR / "train_math.sh").read_text()

    assert "deepscalar_train_formatted.parquet" in script
    assert "DEEPSCALAR_TRAIN_FILE" in script
    assert "math_eval_deepscalar.parquet" in script
    assert "DEEPSCALAR_VAL_FILE" in script


def test_math_script_prefers_formatted_general365_train_data():
    script = (PROJECT_DIR / "train_math.sh").read_text()

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


def test_run_multiple_exp_uses_active_reward_matrix_only():
    script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    assert "[-reward {pdpo|gdpo|new|ori}]" in script
    assert 'REWARDS=("pdpo" "gdpo" "new" "ori")' in script
    assert "pdar-ori" not in script
    assert "pdar|pd|" not in script


def test_pdpo_modes_are_available_and_in_default_matrix():
    math_script = (PROJECT_DIR / "train_math.sh").read_text()
    code_script = (PROJECT_DIR / "train_code.sh").read_text()
    multi_script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    assert "pdpo" in math_script
    assert 'REWARD_LABEL="pdpo"' in math_script
    assert 'ADV_ESTIMATOR="pdpo"' in math_script
    assert 'COMBINE_MODE="pdpo"' in math_script

    assert "pdpo" in code_script
    assert 'REWARD_LABEL="pdpo"' in code_script
    assert 'ADV_ESTIMATOR="pdpo"' in code_script
    assert 'COMBINE_MODE="pdpo"' in code_script

    assert "[-reward {pdpo|gdpo|new|ori}]" in multi_script
    assert 'REWARDS=("pdpo" "gdpo" "new" "ori")' in multi_script
    assert 'pdpo|pdpo_reward)' in multi_script
    assert 'REWARDS=("pdpo")' in multi_script


def test_gdpo_modes_are_available_with_reward_components():
    math_script = (PROJECT_DIR / "train_math.sh").read_text()
    code_script = (PROJECT_DIR / "train_code.sh").read_text()
    multi_script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    math_gdpo = math_script[math_script.index("  gdpo|gdpo_reward)") : math_script.index("  *)", math_script.index("  gdpo|gdpo_reward)"))]
    code_gdpo = code_script[code_script.index("  gdpo|gdpo_reward)") : code_script.index("  *)", code_script.index("  gdpo|gdpo_reward)"))]

    assert 'REWARD_LABEL="gdpo"' in math_gdpo
    assert 'ADV_ESTIMATOR="gdpo"' in math_gdpo
    assert 'COMBINE_MODE="gdpo"' in math_gdpo
    assert 'GDPO_ARGS+=("++algorithm.gdpo_reward_keys=${GDPO_REWARD_KEYS}")' in math_script
    assert 'GDPO_ARGS+=("++algorithm.gdpo_reward_weights=${GDPO_REWARD_WEIGHTS}")' in math_script
    assert "math_step_arithmetic_validity_reward" in math_script
    assert "math_prefix_consistency_reward" in math_script
    assert "math_trace_efficiency_reward" in math_script
    assert "math_answer_extractability_reward" in math_script

    assert 'REWARD_LABEL="gdpo"' in code_gdpo
    assert 'ADV_ESTIMATOR="gdpo"' in code_gdpo
    assert 'COMBINE_MODE="gdpo"' in code_gdpo
    assert 'GDPO_ARGS+=("++algorithm.gdpo_reward_keys=${GDPO_REWARD_KEYS}")' in code_script
    assert 'GDPO_ARGS+=("++algorithm.gdpo_reward_weights=${GDPO_REWARD_WEIGHTS}")' in code_script
    assert "coding_code_extractability_reward" in code_script
    assert "coding_syntax_validity_reward" in code_script
    assert "coding_compiler_runtime_feedback" in code_script

    assert 'gdpo|gdpo_reward)' in multi_script
    assert 'REWARDS=("gdpo")' in multi_script


def test_multi_experiment_runner_uses_renamed_train_scripts():
    script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    assert 'MATH_SCRIPT="${DIR}/train_math.sh"' in script
    assert 'CODE_SCRIPT="${DIR}/train_code.sh"' in script
    assert 'run_grpo_math.sh' not in script
    assert 'run_grpo.sh' not in script


def test_train_scripts_share_save_frequency_interface():
    math_script = (PROJECT_DIR / "train_math.sh").read_text()
    code_script = (PROJECT_DIR / "train_code.sh").read_text()

    for script in (math_script, code_script):
        assert "--save_freq" in script
        assert 'CLI_SAVE_FREQ=""' in script
        assert "CLI_SAVE_FREQ=\"$2\"" in script
        assert "SAVE_EVERY_STEPS=${CLI_SAVE_FREQ:-${SAVE_EVERY_STEPS:--1}}" in script
        assert "MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-5}" in script
        assert "MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-5}" in script
        assert 'trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"' in script
        assert 'trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}"' in script


def test_multi_experiment_runner_labels_code_benchmark_as_eurus():
    script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    assert "BENCHMARK: eurus" in script
    assert "code_eurus" in script
    assert "Code:Eurus" in script
    assert "BENCHMARK: deepcoder" not in script


def test_multi_experiment_runner_uses_deepscalar_as_only_math_dataset():
    script = (PROJECT_DIR / "run_multiple_exp.sh").read_text()

    assert 'MATH_DATASETS=("deepscalar")' in script
    assert '"general365"' not in script
    assert '"gsm8k"' not in script
