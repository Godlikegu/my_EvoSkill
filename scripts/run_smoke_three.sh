#!/usr/bin/env bash
# End-to-end smoke: runs three small tasks in parallel through the harness.
#
# Tasks chosen for shape diversity + small data:
#   - cars_spectroscopy   (1D spectral CARS unmixing)
#   - mri_grappa          (multi-coil MRI parallel imaging)
#   - xray_tooth_gridrec  (3D X-ray CT gridrec reconstruction)
#
# Per-task budget defaults to 1800 s (30 min) and max_rounds=4 so the
# smoke is meaningful without being prohibitively expensive. Override by
# exporting BUDGET_SECONDS / MAX_ROUNDS / MAX_WORKERS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EVOSKILL_ENV_NAME:-evoskill}"
BUDGET_SECONDS="${BUDGET_SECONDS:-1800}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
MAX_WORKERS="${MAX_WORKERS:-3}"

TASKS=(
    "cars_spectroscopy"
    "mri_grappa"
    "xray_tooth_gridrec"
)

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Run as `python -m myevoskill.cli` without requiring `pip install -e .`.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "[smoke] provisioning per-task venvs ..."
for t in "${TASKS[@]}"; do
    bash "${REPO_ROOT}/scripts/setup_task_env.sh" "${t}"
done

echo "[smoke] registering ${#TASKS[@]} tasks ..."
for t in "${TASKS[@]}"; do
    python -m myevoskill.cli register-task \
        --repo-root "${REPO_ROOT}" --task-id "${t}" --force --require-task-env
done

echo "[smoke] running batch with budget=${BUDGET_SECONDS}s rounds=${MAX_ROUNDS} workers=${MAX_WORKERS}"
exec python -m myevoskill.cli run-batch \
    --repo-root "${REPO_ROOT}" \
    --task-ids "${TASKS[@]}" \
    --max-workers "${MAX_WORKERS}" \
    --max-rounds "${MAX_ROUNDS}" \
    --budget-seconds "${BUDGET_SECONDS}"
