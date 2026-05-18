#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SANDBOX_FUSION_ROOT=${SANDBOX_FUSION_ROOT:-"/shared/nas2/yujiz/rl/SandboxFusion"}
CONDA_SH=${CONDA_SH:-"/shared/nas2/yujiz/anaconda3/etc/profile.d/conda.sh"}
SANDBOX_SERVICE_ENV=${SANDBOX_SERVICE_ENV:-"sandbox-service"}
SANDBOX_RUNTIME_ENV=${SANDBOX_RUNTIME_ENV:-"sandbox-runtime"}
SANDBOX_CONFIG_NAME=${SANDBOX_CONFIG_NAME:-"local"}
SANDBOX_HOST=${SANDBOX_HOST:-"127.0.0.1"}
SANDBOX_PORT=${SANDBOX_PORT:-""}
SANDBOX_START_TIMEOUT_S=${SANDBOX_START_TIMEOUT_S:-90}
SANDBOX_STATE_DIR=${SANDBOX_STATE_DIR:-"${SANDBOX_FUSION_ROOT}/.run"}
SANDBOX_LOG_PATH=${SANDBOX_LOG_PATH:-"${SANDBOX_STATE_DIR}/sandbox_fusion.log"}
SANDBOX_PID_FILE=${SANDBOX_PID_FILE:-"${SANDBOX_STATE_DIR}/sandbox_fusion.pid"}
SANDBOX_URL_FILE=${SANDBOX_URL_FILE:-"${SANDBOX_STATE_DIR}/sandbox_fusion.url"}

mkdir -p "${SANDBOX_STATE_DIR}" "${SANDBOX_FUSION_ROOT}/docs/build"

find_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [[ -z "${SANDBOX_PORT}" || "${SANDBOX_PORT}" == "auto" ]]; then
  SANDBOX_PORT=$(find_free_port)
fi

SANDBOX_BASE_URL="http://${SANDBOX_HOST}:${SANDBOX_PORT}"
SANDBOX_FUSION_URL="${SANDBOX_BASE_URL}/run_code"

health_check_sandbox() {
  SANDBOX_BASE_URL="${SANDBOX_BASE_URL}" python3 - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

base = os.environ["SANDBOX_BASE_URL"]

try:
    with urllib.request.urlopen(base + "/v1/ping", timeout=3) as resp:
        if resp.status != 200 or resp.read().decode().strip().strip('"') != "pong":
            sys.exit(1)
except Exception:
    sys.exit(1)

payload = json.dumps(
    {
        "compile_timeout": 5,
        "run_timeout": 5,
        "code": "print('sandbox_healthcheck_ok')",
        "stdin": "",
        "memory_limit_MB": 128,
        "language": "python",
        "files": {},
        "fetch_files": [],
    }
).encode()
req = urllib.request.Request(
    base + "/run_code",
    data=payload,
    headers={"Content-Type": "application/json", "Accept": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
except Exception:
    sys.exit(1)

stdout = (((body.get("run_result") or {}).get("stdout")) or "").strip()
if body.get("status") != "Success" or stdout != "sandbox_healthcheck_ok":
    sys.exit(1)
PY
}

if health_check_sandbox; then
  echo "[INFO] Reusing healthy Sandbox Fusion at ${SANDBOX_FUSION_URL}" >&2
  printf '%s\n' "${SANDBOX_FUSION_URL}" > "${SANDBOX_URL_FILE}"
  printf '%s\n' "${SANDBOX_FUSION_URL}"
  exit 0
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
CONDA_BASE=${CONDA_BASE:-$(conda info --base)}
SANDBOX_SERVICE_PYTHON=${SANDBOX_SERVICE_PYTHON:-"${CONDA_BASE}/envs/${SANDBOX_SERVICE_ENV}/bin/python"}

env_exists() {
  local env_name=$1
  conda env list | awk '{print $1}' | grep -Fxq "${env_name}"
}

ensure_service_env() {
  if env_exists "${SANDBOX_SERVICE_ENV}" && \
     conda run -n "${SANDBOX_SERVICE_ENV}" python -c "import sandbox, uvicorn" >/dev/null 2>&1; then
    return 0
  fi

  echo "[INFO] Bootstrapping Sandbox Fusion service env: ${SANDBOX_SERVICE_ENV}" >&2
  if ! env_exists "${SANDBOX_SERVICE_ENV}"; then
    conda create -n "${SANDBOX_SERVICE_ENV}" -y python=3.11 pip >&2
  fi
  conda run -n "${SANDBOX_SERVICE_ENV}" python -m pip install --upgrade pip setuptools wheel >&2
  conda run -n "${SANDBOX_SERVICE_ENV}" python -m pip install -e "${SANDBOX_FUSION_ROOT}" >&2
}

ensure_runtime_env() {
  if env_exists "${SANDBOX_RUNTIME_ENV}"; then
    return 0
  fi

  echo "[INFO] Bootstrapping Sandbox Fusion runtime env: ${SANDBOX_RUNTIME_ENV}" >&2
  conda create -n "${SANDBOX_RUNTIME_ENV}" -y python=3.10 >&2
}

ensure_service_env
ensure_runtime_env

if [[ ! -x "${SANDBOX_SERVICE_PYTHON}" ]]; then
  echo "[ERROR] Sandbox service python not found: ${SANDBOX_SERVICE_PYTHON}" >&2
  exit 1
fi

echo "[INFO] Starting Sandbox Fusion on ${SANDBOX_BASE_URL}" >&2
echo "[INFO] Logs: ${SANDBOX_LOG_PATH}" >&2

nohup bash -lc "
  cd '${SANDBOX_FUSION_ROOT}' &&
  export TMPDIR='${SANDBOX_STATE_DIR}' &&
  export SANDBOX_CONFIG='${SANDBOX_CONFIG_NAME}' &&
  export PYTHONUNBUFFERED=1 &&
  exec '${SANDBOX_SERVICE_PYTHON}' -m uvicorn sandbox.server.server:app --host '${SANDBOX_HOST}' --port '${SANDBOX_PORT}' --workers 4
" >"${SANDBOX_LOG_PATH}" 2>&1 &
echo $! > "${SANDBOX_PID_FILE}"
printf '%s\n' "${SANDBOX_FUSION_URL}" > "${SANDBOX_URL_FILE}"

deadline=$((SECONDS + SANDBOX_START_TIMEOUT_S))
while (( SECONDS < deadline )); do
  if health_check_sandbox; then
    echo "[INFO] Sandbox Fusion is ready at ${SANDBOX_FUSION_URL}" >&2
    printf '%s\n' "${SANDBOX_FUSION_URL}"
    exit 0
  fi
  sleep 1
done

echo "[ERROR] Sandbox Fusion failed to become healthy within ${SANDBOX_START_TIMEOUT_S}s" >&2
if [[ -f "${SANDBOX_PID_FILE}" ]]; then
  echo "[ERROR] PID: $(cat "${SANDBOX_PID_FILE}")" >&2
fi
if [[ -f "${SANDBOX_LOG_PATH}" ]]; then
  echo "[ERROR] Last Sandbox Fusion log lines:" >&2
  tail -n 80 "${SANDBOX_LOG_PATH}" >&2 || true
fi
exit 1
