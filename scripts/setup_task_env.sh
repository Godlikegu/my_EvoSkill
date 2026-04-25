#!/usr/bin/env bash
# Provision a per-task Python virtualenv from the task's requirements.txt.
#
# Usage:
#     bash MyEvoSkill/scripts/setup_task_env.sh <task_id> [--force] [--python <py>]
#
# What this does:
#   1. Resolves repo_root = directory of this script's parent (MyEvoSkill/).
#   2. Reads `tasks/<task_id>/requirements.txt`. If missing, the task is
#      declared "no extra deps" and the harness Python interpreter is
#      reused (recorded as such in the state file).
#   3. Creates `MyEvoSkill/.venvs/<task_id>/` (idempotent: --force wipes).
#   4. `pip install -r requirements.txt` (online; logs to runtime_logs/).
#   5. Writes `MyEvoSkill/runtime_logs/setup/<task_id>.json` with:
#        {
#          "task_id": "...",
#          "ready": true,
#          "python_executable": "<abs path>",
#          "requirements_path": "<abs path or null>",
#          "requirements_sha256": "<hex or null>",
#          "created_at_unix": <int>,
#          "pip_log": "<abs path>"
#        }
#      The harness's registration step refuses to register a task whose
#      state file is missing or has ready=false (gated behind
#      --require-task-env).
#
# This script is intentionally self-contained: no conda involvement, no
# editable installs, no system-Python pollution.  Each task's venv is
# fully isolated under MyEvoSkill/.venvs/.
set -euo pipefail

usage() {
    cat <<EOF >&2
usage: $0 <task_id> [--force] [--python <python_executable>]

Options:
  --force            wipe and recreate the venv even if it exists
  --python <py>      base interpreter to seed the venv from
                     (default: the python on PATH that matches >=3.9)
EOF
    exit 64
}

if [[ $# -lt 1 ]]; then
    usage
fi

TASK_ID="$1"; shift
FORCE=0
BASE_PY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --python) BASE_PY="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)/tasks"
TASK_DIR="${TASKS_ROOT}/${TASK_ID}"

if [[ ! -d "${TASK_DIR}" ]]; then
    echo "task directory not found: ${TASK_DIR}" >&2
    exit 65
fi

VENVS_ROOT="${REPO_ROOT}/.venvs"
VENV_DIR="${VENVS_ROOT}/${TASK_ID}"
SETUP_LOG_DIR="${REPO_ROOT}/runtime_logs/setup"
mkdir -p "${SETUP_LOG_DIR}" "${VENVS_ROOT}"

STATE_FILE="${SETUP_LOG_DIR}/${TASK_ID}.json"
PIP_LOG="${SETUP_LOG_DIR}/${TASK_ID}.pip.log"
REQ_FILE="${TASK_DIR}/requirements.txt"

# Pick a base interpreter.
if [[ -z "${BASE_PY}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        BASE_PY="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        BASE_PY="$(command -v python)"
    else
        echo "no python on PATH; pass --python <abs path>" >&2
        exit 66
    fi
fi

# Validate Python >= 3.9.
"${BASE_PY}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || {
    echo "${BASE_PY} is older than 3.9; pick a newer interpreter" >&2
    exit 66
}

# (Re)create venv.
if [[ -d "${VENV_DIR}" && "${FORCE}" -eq 1 ]]; then
    echo "[setup_task_env] --force: removing ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[setup_task_env] creating venv: ${VENV_DIR}"
    "${BASE_PY}" -m venv "${VENV_DIR}"
fi

# Locate the venv's python.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PY="${VENV_DIR}/bin/python"
elif [[ -x "${VENV_DIR}/Scripts/python.exe" ]]; then
    VENV_PY="${VENV_DIR}/Scripts/python.exe"
else
    echo "venv python not found inside ${VENV_DIR}" >&2
    exit 70
fi

# Always upgrade pip itself, quietly.
"${VENV_PY}" -m pip install --quiet --upgrade pip wheel setuptools >>"${PIP_LOG}" 2>&1 || {
    echo "pip upgrade failed; see ${PIP_LOG}" >&2
    exit 71
}

# requirements.txt is optional.
REQ_HASH=""
if [[ -f "${REQ_FILE}" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
        REQ_HASH="$(sha256sum "${REQ_FILE}" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        REQ_HASH="$(shasum -a 256 "${REQ_FILE}" | awk '{print $1}')"
    fi
    echo "[setup_task_env] installing ${REQ_FILE}"
    "${VENV_PY}" -m pip install -r "${REQ_FILE}" >>"${PIP_LOG}" 2>&1 || {
        echo "pip install failed; see ${PIP_LOG}" >&2
        # Mark as not ready and exit non-zero.
        "${VENV_PY}" - <<PYEOF >"${STATE_FILE}"
import json, time
print(json.dumps({
    "task_id": "${TASK_ID}",
    "ready": False,
    "error": "pip install failed",
    "python_executable": "${VENV_PY}",
    "requirements_path": "${REQ_FILE}",
    "requirements_sha256": "${REQ_HASH}",
    "created_at_unix": int(time.time()),
    "pip_log": "${PIP_LOG}",
}, indent=2))
PYEOF
        exit 72
    }
else
    echo "[setup_task_env] no requirements.txt; venv is bare"
fi

# Resolve absolute python path inside the venv.
ABS_VENV_PY="$("${VENV_PY}" -c 'import sys; print(sys.executable)')"

"${VENV_PY}" - <<PYEOF >"${STATE_FILE}"
import json, os, time
state = {
    "task_id": "${TASK_ID}",
    "ready": True,
    "python_executable": "${ABS_VENV_PY}",
    "requirements_path": "${REQ_FILE}" if os.path.isfile("${REQ_FILE}") else None,
    "requirements_sha256": "${REQ_HASH}" or None,
    "created_at_unix": int(time.time()),
    "pip_log": "${PIP_LOG}",
}
print(json.dumps(state, indent=2))
PYEOF

echo "[setup_task_env] OK -> ${STATE_FILE}"
echo "                python: ${ABS_VENV_PY}"
