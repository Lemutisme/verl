from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def read_script(name: str) -> str:
    return (PROJECT_DIR / name).read_text()


def test_run_multiple_exp_exposes_and_forwards_method():
    script = read_script("run_multiple_exp.sh")

    assert "-method, --method" in script
    assert 'METHODS=("grpo")' in script
    assert "METHOD_FILTER" in script
    assert '-method "${METHOD}"' in script
    assert "R${ROUND}_math_${DATASET}_${METHOD}_${REWARD}.stdout" in script
    assert "R${ROUND}_code_eurus_${METHOD}_${REWARD}.stdout" in script


def test_child_train_scripts_accept_method():
    for script_name in ("train_math.sh", "train_code.sh"):
        script = read_script(script_name)

        assert "-method, --method" in script
        assert "-method|--method" in script
        assert "METHOD_KIND" in script
        assert "METHOD_LABEL" in script


def test_dapo_method_uses_supported_estimator_and_clip_defaults():
    for script_name in ("train_math.sh", "train_code.sh"):
        script = read_script(script_name)

        assert "dapo)" in script
        assert 'METHOD_ADV_ESTIMATOR_DEFAULT="grpo"' in script
        assert "PPO_CLIP_RATIO_HIGH=${PPO_CLIP_RATIO_HIGH:-0.28}" in script
        assert "PPO_CLIP_RATIO_C=${PPO_CLIP_RATIO_C:-10.0}" in script


def test_multi_run_keeps_best_checkpoint_when_periodic_saves_are_disabled():
    script = read_script("run_multiple_exp.sh")

    assert 'export SAVE_EVERY_STEPS="-1"' in script
    assert 'export SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"' in script
    assert 'export SAVE_BEST_CHECKPOINT="false"' not in script
