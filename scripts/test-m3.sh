#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3-native.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3-reference.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3-numerical.sh"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-m3-package.sh"
