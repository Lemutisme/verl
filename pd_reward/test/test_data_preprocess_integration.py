from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_prepare_data_runs_all_reward_data_repairs():
    script = (PROJECT_DIR / "data_preprocess" / "prepare_data.sh").read_text()

    assert "format_math_prompts.py" in script
    assert "deepscalar_train_formatted.parquet" in script
    assert "general365/train_formatted.parquet" in script
    assert "--force-hash" in script
    assert "clean_deepcoder_data.py" in script
    assert "deepcoder_full_train_clean.parquet" in script
    assert "code_eval_master_clean.parquet" in script
