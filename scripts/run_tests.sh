#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_ENV="${MYEVOSKILL_CONDA_ENV_NAME:-myevoskill}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "A conda environment must be active. Expected: ${EXPECTED_ENV}" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV}" != "${EXPECTED_ENV}" ]]; then
  echo "Active conda environment is '${CONDA_DEFAULT_ENV}', expected '${EXPECTED_ENV}'." >&2
  exit 1
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
pytest "$@"
