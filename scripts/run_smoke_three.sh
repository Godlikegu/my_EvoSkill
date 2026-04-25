#!/usr/bin/env bash
# End-to-end smoke: runs three small tasks in parallel through the harness.
#
# Tasks chosen for shape diversity + small data:
#   - cars_spectroscopy   (1D spectral CARS unmixing)
#   - ct_sparse_view      (2D CT sparse-view reconstruction)
#   - mri_grappa          (multi-coil MRI parallel imaging)
#
# Per-task budget defaults to 600 s and max_rounds=3 so the smoke is cheap
# (a few minutes total against Claude Sonnet, ~few-USD budget). Override by
# exporting BUDGET_SECONDS / MAX_ROUNDS / MAX_WORKERS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EVOSKILL_ENV_NAME:-evoskill}"
BUDGET_SECONDS="${BUDGET_SECONDS:-600}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_WORKERS="${MAX_WORKERS:-3}"

TASKS=(
    "cars_spectroscopy"
    "ct_sparse_view"
    "mri_grappa"
)

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

echo "[smoke] registering ${#TASKS[@]} tasks ..."
for t in "${TASKS[@]}"; do
    python -m myevoskill.cli register-task \
        --repo-root "${REPO_ROOT}" --task-id "${t}" --force
done

echo "[smoke] running batch with budget=${BUDGET_SECONDS}s rounds=${MAX_ROUNDS} workers=${MAX_WORKERS}"
exec python -m myevoskill.cli run-batch \
    --repo-root "${REPO_ROOT}" \
    --task-ids "${TASKS[@]}" \
    --max-workers "${MAX_WORKERS}" \
    --max-rounds "${MAX_ROUNDS}" \
    --budget-seconds "${BUDGET_SECONDS}"
