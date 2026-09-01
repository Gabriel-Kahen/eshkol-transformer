#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command cmp
require_command grep
require_command nm
require_command python3
require_command timeout

l2_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
l2_cc="$(tsv_value "${l2_provenance}" cc_path)"
l2_cxx="$(tsv_value "${l2_provenance}" cxx_path)"
l2_runner="$(eshkol_build_dir)/eshkol-run"
l2_artifact_dir="$(project_build_dir)/l2"
l2_library="${l2_artifact_dir}/libeshkol_transformer_l2.a"
l2_k1_dir="$(project_build_dir)/k1"
l2_k1_library="${l2_k1_dir}/libeshkol_transformer_k1.a"
l2_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-l2.XXXXXX")"
trap 'rm -rf -- "${l2_tmp}"' EXIT
l2_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -I "${PROJECT_ROOT}/include")

[[ -r "${l2_library}" && -r "${l2_artifact_dir}/indexed_cross_entropy.o" ]] ||
  die "canonical L2 archive or object is missing"
[[ "$(ar t "${l2_library}")" == "indexed_cross_entropy.o" ]] ||
  die "canonical L2 archive has unexpected members"
[[ -r "${l2_k1_library}" ]] || die "canonical K1 archive is missing"

nm -g --defined-only "${l2_artifact_dir}/indexed_cross_entropy.o" |
  awk 'NF >= 3 { print $3 }' | sort >"${l2_tmp}/exports.txt"
cmp "${PROJECT_ROOT}/native/l2_public_exports.txt" "${l2_tmp}/exports.txt"
if nm -g "${l2_artifact_dir}/indexed_cross_entropy.o" |
    grep -F "eshkol_transformer_kernel_provider_v1"; then
  die "L2 must not define the canonical K1 provider symbol"
fi
if nm -u "${l2_artifact_dir}/indexed_cross_entropy.o" |
    grep -E '(^|[[:space:]])(malloc|calloc|realloc|free)$'; then
  die "L2 dispatch object must not depend on allocation"
fi
if grep -Ein 'python|pytorch|torch|finite.?difference|autodiff|dlopen|getenv' \
    "${PROJECT_ROOT}/native/indexed_cross_entropy.c" \
    "${PROJECT_ROOT}/include/eshkol_transformer/indexed_cross_entropy.h"; then
  die "L2 production ABI contains a forbidden runtime/fallback reference"
fi

"${l2_cc}" "${l2_cflags[@]}" \
  "${PROJECT_ROOT}/tests/l2/test_indexed_cross_entropy.c" \
  "${l2_library}" "${l2_k1_library}" -lm -o "${l2_tmp}/test-l2"
for l2_run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${l2_tmp}/test-l2" >"${l2_tmp}/native-${l2_run}.stdout"
done
cmp "${l2_tmp}/native-1.stdout" "${l2_tmp}/native-2.stdout"
grep -E '^L2 PASS: [0-9]+ capability, numerical, gradient, and adversarial checks$' \
  "${l2_tmp}/native-1.stdout" >/dev/null

"${l2_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/tests/l2/header_cpp.cpp" \
  "${l2_library}" "${l2_k1_library}" -lm -o "${l2_tmp}/header-cpp"
"${l2_tmp}/header-cpp"

"${l2_cc}" "${l2_cflags[@]}" "${PROJECT_ROOT}/tests/l2/report.c" \
  "${l2_library}" "${l2_k1_library}" -lm -o "${l2_tmp}/report"
LC_ALL=C "${l2_tmp}/report" >"${l2_tmp}/report.c.json"
LC_ALL=C.UTF-8 "${l2_tmp}/report" >"${l2_tmp}/report.c-utf8.json"
cmp "${l2_tmp}/report.c.json" "${l2_tmp}/report.c-utf8.json"
python3 "${PROJECT_ROOT}/tests/l2/validate_report.py" \
  "${l2_tmp}/report.c.json"

"${l2_cc}" "${l2_cflags[@]}" "${PROJECT_ROOT}/tests/l2/oracle_runner.c" \
  "${l2_library}" "${l2_k1_library}" -lm -o "${l2_tmp}/oracle-runner"
PYTHONPATH="${PROJECT_ROOT}" PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/l2/test_oracle_parity.py" \
  "${l2_tmp}/oracle-runner"

"${l2_cc}" "${l2_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/l2/eshkol_bridge.c" -o "${l2_tmp}/eshkol_bridge.o"
ar rcsD "${l2_tmp}/libeshkol_transformer_l2_eshkol_test.a" \
  "${l2_tmp}/eshkol_bridge.o" \
  "${l2_artifact_dir}/indexed_cross_entropy.o" \
  "${l2_k1_dir}/kernel_abi.o"
for l2_run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${l2_runner}" --strict-types --no-stdlib \
    -L "${l2_tmp}" --lib eshkol_transformer_l2_eshkol_test \
    "${PROJECT_ROOT}/tests/l2/eshkol_runtime.esk" \
    -o "${l2_tmp}/eshkol-${l2_run}" \
    >"${l2_tmp}/eshkol-compile-${l2_run}.stdout" \
    2>"${l2_tmp}/eshkol-compile-${l2_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s \
    "${l2_tmp}/eshkol-${l2_run}" \
    >"${l2_tmp}/eshkol-${l2_run}.stdout" \
    2>"${l2_tmp}/eshkol-${l2_run}.stderr"
  cmp "${PROJECT_ROOT}/tests/l2/expected/eshkol.stdout" \
    "${l2_tmp}/eshkol-${l2_run}.stdout"
  test ! -s "${l2_tmp}/eshkol-${l2_run}.stderr"
done
cmp "${l2_tmp}/eshkol-1.stdout" "${l2_tmp}/eshkol-2.stdout"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-k1.sh" \
  "${l2_tmp}/sanitized-k1" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-l2.sh" \
  "${l2_tmp}/sanitized-l2" sanitize
"${l2_cc}" "${l2_cflags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/l2/test_indexed_cross_entropy.c" \
  "${l2_tmp}/sanitized-l2/libeshkol_transformer_l2.a" \
  "${l2_tmp}/sanitized-k1/libeshkol_transformer_k1.a" -lm \
  -o "${l2_tmp}/test-l2-sanitized"
ASAN_OPTIONS=detect_leaks="${L2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${l2_tmp}/test-l2-sanitized" >/dev/null

printf 'L2 PASS: explicit provider, f32 forward/direct-backward parity, negatives, failure atomicity, Eshkol AOT, isolation, and sanitizers\n'
