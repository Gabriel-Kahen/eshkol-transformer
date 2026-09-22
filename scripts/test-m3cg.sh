#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
python3 -m unittest -v tests.m3cg.test_contract
python3 "${PROJECT_ROOT}/tests/m3cg/measure_gate.py" "$(project_build_dir)/m3cg"
