#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.conda_env"

echo "Creating MyEvoSkill dev environment at ${ENV_PREFIX}"
conda env create -p "${ENV_PREFIX}" -f "${ROOT_DIR}/environment.yml" || conda env update -p "${ENV_PREFIX}" -f "${ROOT_DIR}/environment.yml"
echo "Done. Activate with:"
echo "  conda activate ${ENV_PREFIX}"

