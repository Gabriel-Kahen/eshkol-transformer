#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_exec="${Q0_PYTHON:-${project_root}/.tmp/q0-venv/bin/python}"

if [[ ! -x "${python_exec}" ]]; then
  printf 'error: Q0 Python environment unavailable: %s\n' "${python_exec}" >&2
  printf 'install tests/q0/requirements-oracle.lock or set Q0_PYTHON\n' >&2
  exit 1
fi
if [[ -z "${ESHKOL_RUN:-}" ]]; then
  printf 'error: set ESHKOL_RUN to the canonical executable eshkol-run path\n' >&2
  exit 1
fi
if [[ ! -x "${ESHKOL_RUN}" ]]; then
  printf 'error: ESHKOL_RUN is not executable: %s\n' "${ESHKOL_RUN}" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${TMPDIR:-${project_root}/.tmp/q0-test-tmp}"
mkdir -p -- "${TMPDIR}"

cd -- "${project_root}"
"${python_exec}" -m unittest discover -s tests/q0 -p 'test_*.py' -v
