#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
require_command python3

output="${1:-$(project_build_dir)/e3-diagnostic-private-gate}"
mkdir -p "${output}"

(cd "${PROJECT_ROOT}" && python3 -m unittest -q \
  tests.e3_diagnostic_private.test_source_contract \
  tests.e3_diagnostic_private.test_package_policy \
  tests.ci.test_e3_metrics_registration)

# Rebuild and test the unchanged base tuple first. This is the supported
# byte-exact regression guard for the shared two-tuple policy change.
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-private.sh" "${output}/base"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${output}/base"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-e3-private-runtime.sh" \
  "${output}/base" "${output}/base-runtime"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-diagnostic-private.sh" \
  "${output}/diagnostic"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-e3-diagnostic-private-package.sh" \
  "${output}/diagnostic"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-e3-diagnostic-private-runtime.sh" \
  "${output}/diagnostic" "${output}/diagnostic-runtime"

printf 'E3 diagnostic private successor gate passed: %s\n' "${output}"
