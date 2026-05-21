from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parents[0]


def test_legacy_pd_and_pdar_files_are_removed():
    removed_paths = [
        PROJECT_DIR / "pdar_advantage.py",
        PROJECT_DIR / "pdar_init.py",
        PROJECT_DIR / "reward_score" / "pdar_core.py",
        PROJECT_DIR / "reward_score" / "primal_dual_core.py",
        PROJECT_DIR / "test" / "test_pdar_advantage.py",
        PROJECT_DIR / "test" / "test_pdar_core.py",
        PROJECT_DIR / "test" / "test_pd_fixes.py",
    ]

    assert (PROJECT_DIR / "pdpo_init.py").is_file()
    for path in removed_paths:
        assert not path.exists(), f"legacy PD/PDAR file still exists: {path}"


def test_trainer_registers_only_pdpo_custom_estimator():
    trainer = (REPO_DIR / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text()

    assert "pdpo_init" in trainer
    assert "pdar_init" not in trainer
    assert "PDAR_METRICS" not in trainer
    assert "pdar_config_dict" not in trainer


def test_pdpo_advantage_owns_group_norm_and_damping_helpers():
    source = (PROJECT_DIR / "pdpo_advantage.py").read_text()

    assert "def group_normalize_scores" in source
    assert "def selective_damp" in source
    assert "reward_score.pdar_core" not in source
    assert "pdar_" not in source


def test_custom_reward_rejects_legacy_pd_and_pdar_modes():
    import custom_reward

    assert custom_reward._normalize_combine_mode("pdpo") == "pdpo"
    assert custom_reward._normalize_combine_mode("new") == "multiplier"

    for mode in ("pd", "pdar", "pdar-ori"):
        with pytest.raises(ValueError, match="Unsupported combine_mode"):
            custom_reward._normalize_combine_mode(mode)
