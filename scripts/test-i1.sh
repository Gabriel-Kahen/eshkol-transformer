#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command cmp
require_command grep
require_command timeout

i1_unsupported_tool_pattern='(^|[[:space:]])r[g]([[:space:]]|$)'
printf '%s\n' 'rg --version' | grep -Eq "${i1_unsupported_tool_pattern}" || \
  die "I1 unsupported-tool regression probe is ineffective"
if grep -En "${i1_unsupported_tool_pattern}" \
    "${PROJECT_ROOT}/scripts/build-i1.sh" \
    "${PROJECT_ROOT}/scripts/test-i1.sh"; then
  die "I1 scripts must use supported-lane baseline tools, not ripgrep"
fi

i1_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
i1_cc="$(tsv_value "${i1_provenance}" cc_path)"
i1_cxx="$(tsv_value "${i1_provenance}" cxx_path)"
i1_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-i1.XXXXXX")"
trap 'rm -rf -- "${i1_tmp}"' EXIT
i1_artifact_dir="$(project_build_dir)/i1"
i1_library="${i1_artifact_dir}/libeshkol_transformer_i64.a"
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
i1_runner="$(eshkol_build_dir)/eshkol-run"

i1_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fPIC -I "${PROJECT_ROOT}/include"
)

[[ -r "${i1_library}" && -r "${i1_artifact_dir}/i64_tensor.o" ]] || \
  die "canonical I1 archive or object is missing"
[[ "$(ar t "${i1_library}")" == "i64_tensor.o" ]] || \
  die "canonical I1 archive has unexpected members"
[[ -r "${k1_library}" ]] || die "canonical K1 archive is missing"

"${i1_cc}" "${i1_cflags[@]}" \
  "${PROJECT_ROOT}/tests/i1/test_i64_tensor.c" \
  "${i1_library}" "${k1_library}" -o "${i1_tmp}/test-i64-tensor"
for i1_run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${i1_tmp}/test-i64-tensor" >"${i1_tmp}/native-${i1_run}.stdout"
done
cmp "${i1_tmp}/native-1.stdout" "${i1_tmp}/native-2.stdout"
grep -E '^I1 PASS: [0-9]+ exact-i64 boundary, ownership, and K1 checks$' \
  "${i1_tmp}/native-1.stdout" >/dev/null

"${i1_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/tests/i1/header_cpp.cpp" \
  "${i1_library}" "${k1_library}" -o "${i1_tmp}/header-cpp"
"${i1_tmp}/header-cpp"

"${i1_cc}" "${i1_cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/i1/interop_bridge.c" \
  -o "${i1_tmp}/interop_bridge.o"
ar rcsD "${i1_tmp}/libeshkol_transformer_i1_interop.a" \
  "${i1_tmp}/interop_bridge.o" \
  "${i1_artifact_dir}/i64_tensor.o" \
  "$(project_build_dir)/k1/kernel_abi.o"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${i1_runner}" --strict-types --emit-object \
  --emit-depfile "${i1_tmp}/interop.d" --no-stdlib \
  -I "${PROJECT_ROOT}/lib" \
  -I "${PROJECT_ROOT}/tests/fixtures/i1" \
  "${PROJECT_ROOT}/tests/i1/interop.esk" \
  -o "${i1_tmp}/interop.o" \
  >"${i1_tmp}/interop-object.stdout" \
  2>"${i1_tmp}/interop-object.stderr"
grep -F 'tests/fixtures/i1/transformer/i64_tensor_internal.esk' \
  "${i1_tmp}/interop.d" >/dev/null
grep -F 'lib/transformer/error_internal.esk' \
  "${i1_tmp}/interop.d" >/dev/null
for i1_run in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 60s \
    "${i1_runner}" --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/tests/fixtures/i1" \
    -L "${i1_tmp}" --lib eshkol_transformer_i1_interop \
    "${PROJECT_ROOT}/tests/i1/interop.esk" \
    -o "${i1_tmp}/interop-${i1_run}" \
    >"${i1_tmp}/interop-compile-${i1_run}.stdout" \
    2>"${i1_tmp}/interop-compile-${i1_run}.stderr"
  timeout --foreground --signal=TERM --kill-after=5s 30s \
    "${i1_tmp}/interop-${i1_run}" \
    >"${i1_tmp}/interop-${i1_run}.stdout" \
    2>"${i1_tmp}/interop-${i1_run}.stderr"
  cmp "${PROJECT_ROOT}/tests/i1/expected/interop.stdout" \
    "${i1_tmp}/interop-${i1_run}.stdout"
  test ! -s "${i1_tmp}/interop-${i1_run}.stderr"
done
cmp "${i1_tmp}/interop-1.stdout" "${i1_tmp}/interop-2.stdout"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-k1.sh" \
  "${i1_tmp}/sanitized-k1" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" \
  "${i1_tmp}/sanitized-i1" sanitize-test
"${i1_cc}" "${i1_cflags[@]}" -DET_I64_TENSOR_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/i1/test_i64_tensor.c" \
  "${i1_tmp}/sanitized-i1/libeshkol_transformer_i64.a" \
  "${i1_tmp}/sanitized-k1/libeshkol_transformer_k1.a" \
  -o "${i1_tmp}/test-i64-tensor-sanitized"
ASAN_OPTIONS=detect_leaks="${I1_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${i1_tmp}/test-i64-tensor-sanitized" >/dev/null

if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/i64_tensor.c" \
    "${PROJECT_ROOT}/include/eshkol_transformer/i64_tensor.h"; then
  die "I1 production path contains a forbidden Python/PyTorch reference"
fi
test ! -e "${PROJECT_ROOT}/lib/transformer/i64_tensor_internal.esk" || \
  die "test-only I1 Eshkol facade leaked into the production module root"

printf 'I1 PASS: C/C++ ABI, exact-i64 storage, K1 copy provider, ownership, canonical Eshkol AOT, determinism, and sanitizers\n'
