#!/usr/bin/env bash
# Create / refresh the `evoskill` conda environment from environment.yml.
#
# Usage:  bash MyEvoSkill/scripts/setup_env.sh
#
# Idempotent: if the env already exists we run `conda env update`.
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

conda activate "${ENV_NAME}"
echo "[setup_env] installing MyEvoSkill in editable mode ..."
pip install -e "${REPO_ROOT}"

echo "[setup_env] OK. Activate with:  conda activate ${ENV_NAME}"
