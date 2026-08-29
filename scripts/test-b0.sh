#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command python3
python3 -m py_compile "${PROJECT_ROOT}"/tests/b0/*.py
python3 -m unittest discover -s "${PROJECT_ROOT}/tests/b0" -p 'test_*.py' -v
