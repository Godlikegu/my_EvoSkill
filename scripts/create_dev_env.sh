#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${MYEVOSKILL_CONDA_ENV_NAME:-${1:-myevoskill}}"

echo "Creating or updating MyEvoSkill dev environment '${ENV_NAME}'"
conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yml" || conda env update -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yml"
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
