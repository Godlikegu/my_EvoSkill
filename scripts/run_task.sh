#!/usr/bin/env bash
# Run a single registered task end-to-end.
#
# Usage:
#   bash MyEvoSkill/scripts/run_task.sh <task_id> [extra args forwarded to cli]
#
# Examples:
#   bash MyEvoSkill/scripts/run_task.sh cars_spectroscopy
#   bash MyEvoSkill/scripts/run_task.sh ct_sparse_view --max-rounds 3 --budget-seconds 600
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <task_id> [cli args]" >&2
    exit 64
fi

TASK_ID="$1"; shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EVOSKILL_ENV_NAME:-evoskill}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

# Sensible defaults; user may override by passing the same flag again.
exec python -m myevoskill.cli run-task \
    --repo-root "${REPO_ROOT}" \
    --task-id "${TASK_ID}" \
    "$@"
