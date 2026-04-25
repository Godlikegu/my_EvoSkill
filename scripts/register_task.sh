#!/usr/bin/env bash
# Register / refresh a task manifest.
#
# Usage:  bash MyEvoSkill/scripts/register_task.sh <task_id> [--force]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <task_id> [--force]" >&2
    exit 64
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EVOSKILL_ENV_NAME:-evoskill}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"
python -m myevoskill.cli register-task --repo-root "${REPO_ROOT}" --task-id "$@"
