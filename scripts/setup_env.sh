#!/usr/bin/env bash
# Create / refresh the `evoskill` conda environment from environment.yml.
#
# Usage:  bash MyEvoSkill/scripts/setup_env.sh
#
# Idempotent: if the env already exists we run `conda env update`.
#
# Note: MyEvoSkill is intentionally NOT installed as a package (no
# pyproject.toml, no `pip install -e .`). All entry points are run as
# `python -m myevoskill.cli ...` with `PYTHONPATH=<repo>/src` prepended.
# The companion scripts (register_task.sh / run_task.sh / run_smoke_three.sh)
# do that for you. If you launch `python -m myevoskill.cli` by hand, run:
#
#     export PYTHONPATH="<repo>/MyEvoSkill/src:${PYTHONPATH:-}"
#
# Per-task runtime dependencies live in their OWN venvs under
# `MyEvoSkill/.venvs/<task_id>/` and are provisioned by
# `scripts/setup_task_env.sh <task_id>`. Conda is only used for this
# harness env (`evoskill`).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="${EVOSKILL_ENV_NAME:-evoskill}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH; install miniconda/anaconda first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[setup_env] updating existing env '${ENV_NAME}' ..."
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo "[setup_env] creating env '${ENV_NAME}' from ${ENV_FILE} ..."
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

cat <<EOF
[setup_env] OK.

Activate with:   conda activate ${ENV_NAME}
Then export:     export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
                 (the helper scripts in MyEvoSkill/scripts/ do this for you)

Three-step pipeline per task:
  bash ${REPO_ROOT}/scripts/setup_task_env.sh <task_id>
  bash ${REPO_ROOT}/scripts/register_task.sh <task_id>
  bash ${REPO_ROOT}/scripts/run_task.sh      <task_id>
EOF
