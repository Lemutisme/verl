from pathlib import Path
import re


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _ray_tmpdir_default(script_name: str) -> str:
    script_text = (PROJECT_DIR / script_name).read_text()
    match = re.search(
        r'^RAY_TMP_ROOT=.*\n'
        r'RAY_TMP_TAG=.*\n'
        r'RAY_TMPDIR=\$\{RAY_TMPDIR:-"(?P<default>[^"]+)"\}',
        script_text,
        re.MULTILINE,
    )
    assert match is not None, f"Could not find RAY_TMPDIR default in {script_name}"
    return match.group("default")


def _ray_tmp_root_default(script_name: str) -> str:
    script_text = (PROJECT_DIR / script_name).read_text()
    match = re.search(r'^RAY_TMP_ROOT=\$\{RAY_TMP_ROOT:-"(?P<default>[^"]+)"\}', script_text, re.MULTILINE)
    assert match is not None, f"Could not find RAY_TMP_ROOT default in {script_name}"
    return match.group("default")


def test_ray_tmpdir_default_is_scoped_to_each_run():
    for script_name in ("train_math.sh", "train_code.sh"):
        assert _ray_tmpdir_default(script_name) == "${RAY_TMP_ROOT}/${RAY_TMP_TAG}"


def test_ray_tmp_root_default_avoids_full_system_tmp():
    for script_name in ("train_math.sh", "train_code.sh"):
        root = _ray_tmp_root_default(script_name)
        assert root == "/dev/shm/ray_yujiz"
        assert not root.startswith("/tmp/")

    run_multiple = (PROJECT_DIR / "run_multiple_exp.sh").read_text()
    assert 'export RAY_TMP_ROOT="${RAY_TMP_ROOT:-/dev/shm/ray_yujiz}"' in run_multiple


def test_ray_tmp_root_default_keeps_ray_socket_paths_short():
    session_tail = "session_2026-06-01_00-04-18_109950_4016739/sockets/plasma_store"
    tag = "0601000404_4015947"

    for script_name in ("train_math.sh", "train_code.sh"):
        root = _ray_tmp_root_default(script_name)
        socket_path = f"{root}/{tag}/{session_tail}"
        assert len(socket_path) <= 107
