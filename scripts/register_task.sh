#!/usr/bin/env bash
# Register / refresh a task manifest.
#
# Usage:
#   bash MyEvoSkill/scripts/register_task.sh <task_id> [--force] [--no-task-env]
#
# By default this script first builds (or verifies) the per-task venv via
# scripts/setup_task_env.sh, then asks the registrar to refuse if the venv
# is not ready. Pass ``--no-task-env`` to skip the venv build (useful for
# tests, or for tasks that need no extra deps and are happy with the
# harness Python).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <task_id> [--force] [--no-task-env]" >&2
    exit 64
fi

TASK_ID="$1"
shift

SKIP_TASK_ENV=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --no-task-env) SKIP_TASK_ENV=1 ;;
        *) EXTRA_ARGS+=("${arg}") ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EVOSKILL_ENV_NAME:-evoskill}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Run as `python -m myevoskill.cli` without requiring `pip install -e .`.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

if [[ "${SKIP_TASK_ENV}" -eq 0 ]]; then
    bash "${REPO_ROOT}/scripts/setup_task_env.sh" "${TASK_ID}"
    REQUIRE_FLAG=(--require-task-env)
else
    REQUIRE_FLAG=()
fi

python -m myevoskill.cli register-task \
    --repo-root "${REPO_ROOT}" \
    --task-id "${TASK_ID}" \
    "${REQUIRE_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"
