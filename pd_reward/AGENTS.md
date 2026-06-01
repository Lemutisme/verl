# Repository Guidelines

## Project Structure & Module Organization

This directory contains PDPO reward and advantage experiments within the larger `verl` repository. Core entry points live at the top level: `custom_reward.py`, `pdpo_advantage.py`, and `pdpo_init.py`. Reward implementations and shared helpers live in `reward_score/`, with subreward utilities under `reward_score/sub_reward/`. Dataset preparation scripts are in `data_preprocess/`. Experiment launchers are shell scripts such as `train_math.sh`, `train_code.sh`, and `run_multiple_exp.sh`. Unit tests live in `test/` as `test_*.py`. Generated outputs such as `logs_*` are ignored and should not be committed.

## Build, Test, and Development Commands

Use the root repository Python environment setup from `../AGENTS.md`. Common local commands:

```bash
pytest -q
pytest -q test/test_pdpo_advantage.py
python -m compileall -q .
bash train_math.sh -reward pdpo -dataset gsm8k -gpus 5
bash run_multiple_exp.sh -gpus 5 -reward pdpo
```

`pytest -q` runs the local test suite. Targeted pytest commands are preferred while iterating on one estimator. `compileall` catches syntax errors without launching experiments. Shell launchers start training or sweep jobs and may require GPUs, Ray, vLLM, and dataset access.

## Coding Style & Naming Conventions

Follow the root `pyproject.toml`: Python uses Ruff formatting and linting with a 120-character line length. Use 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for environment variables. Keep reward-channel names explicit, for example `math_step_arithmetic_validity_reward`, because trainer-side extraction depends on stable keys.

## Testing Guidelines

Add or update `test/test_*.py` files for estimator, reward, and script-behavior changes. Prefer small deterministic tests over GPU-dependent integration tests. Reset global PDPO state in fixtures when tests mutate estimator state. For reward-channel changes, cover flat main-reward groups, tied groups, zero-variance auxiliary channels, and anti-correlated auxiliary signals.

## Commit & Pull Request Guidelines

Recent history uses concise conventional-style subjects such as `feat: ...`, `fix: ...`, and `chore: ...`; match that style. Before proposing a PR, follow the duplicate-work checks and accountability requirements in `../AGENTS.md`. PR descriptions must list tests run, explain why the change is not duplicative, and state when AI assistance was used.

## Configuration & Experiment Hygiene

Keep defaults explicit in scripts and document new environment variables in `README.md`. Avoid committing logs, checkpoints, local datasets, credentials, or machine-specific paths. When changing launch scripts, preserve existing flags unless the behavior change is intentional and tested.
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 69
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/shared/nas2/yujiz/rl/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/shared/nas2/yujiz/rl/verl/pd_reward/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /shared/nas2/yujiz/rl/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/shared/nas2/yujiz/rl/verl/pd_reward" --reconcile`
For copied Codex installs, use:
`bash /shared/nas2/yujiz/rl/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/shared/nas2/yujiz/rl/verl/pd_reward"`
<!-- ARIS-CODEX:END -->
