from pathlib import Path
import re


SCRIPT_DIR = Path(__file__).resolve().parent


def _ray_tmpdir_default(script_name: str) -> str:
    script_text = (SCRIPT_DIR / script_name).read_text()
    match = re.search(
        r'^RAY_TMP_ROOT=.*\n'
        r'RAY_TMP_TAG=.*\n'
        r'RAY_TMPDIR=\$\{RAY_TMPDIR:-"(?P<default>[^"]+)"\}',
        script_text,
        re.MULTILINE,
    )
    assert match is not None, f"Could not find RAY_TMPDIR default in {script_name}"
    return match.group("default")


def test_ray_tmpdir_default_is_scoped_to_each_run():
    for script_name in ("run_grpo_math.sh", "run_grpo.sh"):
        assert _ray_tmpdir_default(script_name) == "${RAY_TMP_ROOT}/${RAY_TMP_TAG}"
