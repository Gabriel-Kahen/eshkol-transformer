#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
test_root="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-test.XXXXXX")"
trap 'rm -rf -- "${test_root}"' EXIT

for run in 1 2; do
  output_dir="${test_root}/run-${run}"
  /usr/bin/bash "${PROJECT_ROOT}/scripts/compile-smoke.sh" "${output_dir}" \
    >"${test_root}/compile-${run}.stdout" 2>"${test_root}/compile-${run}.stderr"
  (cd -- "${PROJECT_ROOT}" && "${output_dir}/eshkol-transformer-smoke") \
    >"${test_root}/run-${run}.stdout" 2>"${test_root}/run-${run}.stderr"
  cmp --silent "${PROJECT_ROOT}/tests/expected/smoke.stdout" "${test_root}/run-${run}.stdout" || \
    die "positive smoke run ${run} output mismatch"
done
cmp --silent "${test_root}/run-1.stdout" "${test_root}/run-2.stdout" || \
  die "fresh repeated smoke outputs differ"
printf 'positive smoke: PASS (two fresh AOT compilations and executions)\n'
printf 'deterministic output: PASS (byte-identical including newline)\n'

missing_log="${test_root}/missing-toolchain.log"
if ESHKOL_SOURCE_DIR="${test_root}/missing-source" \
    ESHKOL_BUILD_DIR="${test_root}/missing-build" \
    BUILD_DIR="${test_root}/missing-output" \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/configure.sh" >"${missing_log}" 2>&1; then
  die "missing-toolchain configure unexpectedly succeeded"
fi
grep -F 'Eshkol source checkout not found' "${missing_log}" >/dev/null || \
  die "missing-toolchain failure did not provide the expected actionable diagnostic"
printf 'missing toolchain: PASS (nonzero with actionable diagnostic)\n'
