#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command python3
require_command sha256sum
require_command timeout

K1_PROVENANCE="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
K1_CC="$(tsv_value "${K1_PROVENANCE}" cc_path)"
K1_CXX="$(tsv_value "${K1_PROVENANCE}" cxx_path)"
K1_RUNNER="$(eshkol_build_dir)/eshkol-run"
K1_TMP="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-k1.XXXXXX")"
trap 'rm -rf -- "${K1_TMP}"' EXIT

K1_CFLAGS=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fPIC -I "${PROJECT_ROOT}/include"
)

"${K1_CC}" "${K1_CFLAGS[@]}" -c \
  "${PROJECT_ROOT}/native/kernel_abi.c" -o "${K1_TMP}/kernel_abi.o"
ar rcs "${K1_TMP}/libeshkol_transformer_k1.a" "${K1_TMP}/kernel_abi.o"

"${K1_CC}" "${K1_CFLAGS[@]}" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  "${PROJECT_ROOT}/tests/k1/test_kernel_abi.c" \
  -o "${K1_TMP}/test-kernel-abi"
for K1_RUN in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${K1_TMP}/test-kernel-abi" >"${K1_TMP}/test-${K1_RUN}.stdout"
done
cmp "${K1_TMP}/test-1.stdout" "${K1_TMP}/test-2.stdout"
grep -E '^K1 PASS: [0-9]+ ABI, discovery, report, and malformed-call checks$' \
  "${K1_TMP}/test-1.stdout" >/dev/null

"${K1_CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" \
  "${PROJECT_ROOT}/tests/k1/header_cpp.cpp" \
  "${K1_TMP}/kernel_abi.o" -o "${K1_TMP}/header-cpp"
"${K1_TMP}/header-cpp"

"${K1_CC}" "${K1_CFLAGS[@]}" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  "${PROJECT_ROOT}/tests/k1/report_baseline.c" \
  -o "${K1_TMP}/report-baseline"
LC_ALL=C "${K1_TMP}/report-baseline" >"${K1_TMP}/baseline_report_v1.json"
LC_ALL=C.UTF-8 "${K1_TMP}/report-baseline" \
  >"${K1_TMP}/baseline_report_v1.c-utf8.json"
cmp "${K1_TMP}/baseline_report_v1.json" \
  "${K1_TMP}/baseline_report_v1.c-utf8.json"
(
  cd -- "${K1_TMP}"
  sha256sum -c "${PROJECT_ROOT}/tests/k1/expected/baseline_report_v1.sha256"
)
python3 "${PROJECT_ROOT}/tests/k1/validate_report.py" \
  "${K1_TMP}/baseline_report_v1.json"

for K1_RUN in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${K1_RUNNER}" --no-stdlib \
    -L "${K1_TMP}" --lib eshkol_transformer_k1 \
    "${PROJECT_ROOT}/tests/k1/abi_version.esk" \
    -o "${K1_TMP}/abi-version-${K1_RUN}" \
    >"${K1_TMP}/eshkol-compile-${K1_RUN}.stdout" \
    2>"${K1_TMP}/eshkol-compile-${K1_RUN}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s \
    "${K1_TMP}/abi-version-${K1_RUN}" \
    >"${K1_TMP}/eshkol-run-${K1_RUN}.stdout" \
    2>"${K1_TMP}/eshkol-run-${K1_RUN}.stderr"
  cmp "${PROJECT_ROOT}/tests/k1/expected/abi_version.stdout" \
    "${K1_TMP}/eshkol-run-${K1_RUN}.stdout"
  test ! -s "${K1_TMP}/eshkol-run-${K1_RUN}.stderr"
done
cmp "${K1_TMP}/eshkol-run-1.stdout" "${K1_TMP}/eshkol-run-2.stdout"

"${K1_CC}" "${K1_CFLAGS[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${PROJECT_ROOT}/native/kernel_abi.c" \
  "${PROJECT_ROOT}/tests/k1/test_kernel_abi.c" \
  -o "${K1_TMP}/test-kernel-abi-sanitized"
ASAN_OPTIONS=detect_leaks="${K1_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  "${K1_TMP}/test-kernel-abi-sanitized" >/dev/null

printf 'K1 PASS: C/C++ ABI, compatible-minor discovery, canonical reports, negatives, sanitizers, and Eshkol AOT link\n'
