#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain

"$(eshkol_build_dir)/eshkol-run" --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/internal/t1/lib" \
  -I "${PROJECT_ROOT}/internal/t2/lib" \
  -I "${PROJECT_ROOT}/internal/data4/lib" \
  -r "${PROJECT_ROOT}/tests/data4/data4_runtime.esk"
