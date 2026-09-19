#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
m3_oracle="${M3_ORACLE_PYTHON:-${N2_ORACLE_PYTHON:-}}"
[[ "${m3_oracle}" == /* && -x "${m3_oracle}" ]] || die "M3_ORACLE_PYTHON must name an absolute pinned Python 3.14.6/PyTorch 2.13.0+cpu oracle interpreter or wrapper"
cd "${PROJECT_ROOT}"
export ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE
"${m3_oracle}" -m unittest tests.m3.test_reference tests.m3.test_public_parity tests.m3.test_native_retention -v
mkdir -p "$(project_build_dir)/m3-reference"
"${m3_oracle}" -m tests.m3.reference --output "$(project_build_dir)/m3-reference/development.json"
"${m3_oracle}" -m tests.m3.reference --output "$(project_build_dir)/m3-reference/development-repeat.json"
cmp "$(project_build_dir)/m3-reference/development.json" "$(project_build_dir)/m3-reference/development-repeat.json"
