#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command cmp
require_command grep
require_command nm
require_command timeout

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-i2.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT
artifact_dir="$(project_build_dir)/i2"
library="${artifact_dir}/libeshkol_transformer_f32.a"
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)

[[ -r "${library}" && -r "${artifact_dir}/f32_tensor.o" ]] || \
  die "canonical I2 archive or object is missing"
[[ "$(ar t "${library}")" == "f32_tensor.o" ]] || \
  die "canonical I2 archive has unexpected members"
[[ -r "${k1_library}" ]] || die "canonical K1 archive is missing"

for source in test_f32_tensor test_f32_parameter; do
  "${cc}" "${cflags[@]}" "${PROJECT_ROOT}/tests/i2/${source}.c" \
    "${library}" "${k1_library}" -o "${temporary_dir}/${source}"
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 90s \
      "${temporary_dir}/${source}" >"${temporary_dir}/${source}-${run}.stdout"
  done
  cmp "${temporary_dir}/${source}-1.stdout" \
    "${temporary_dir}/${source}-2.stdout"
done
grep -E '^I2 storage PASS: [0-9]+ ' \
  "${temporary_dir}/test_f32_tensor-1.stdout" >/dev/null
grep -E '^I2 parameter PASS: [0-9]+ ' \
  "${temporary_dir}/test_f32_parameter-1.stdout" >/dev/null

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/tests/i2/header_cpp.cpp" \
  "${library}" "${k1_library}" -o "${temporary_dir}/header-cpp"
"${temporary_dir}/header-cpp"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/i2/parameter_header_cpp.cpp" \
  "${library}" "${k1_library}" -o "${temporary_dir}/parameter-header-cpp"
"${temporary_dir}/parameter-header-cpp"

if nm -g --defined-only --format=posix "${library}" | \
    grep -E '^eshkol_transformer_kernel_provider_v1[[:space:]]'; then
  die "I2 archive defines the forbidden canonical K1 provider symbol"
fi

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-k1.sh" \
  "${temporary_dir}/sanitized-k1" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i2.sh" \
  "${temporary_dir}/sanitized-i2" sanitize-test
for source in test_f32_tensor test_f32_parameter; do
  "${cc}" "${cflags[@]}" -DET_F32_TENSOR_TESTING \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "${PROJECT_ROOT}/tests/i2/${source}.c" \
    "${temporary_dir}/sanitized-i2/libeshkol_transformer_f32.a" \
    "${temporary_dir}/sanitized-k1/libeshkol_transformer_k1.a" \
    -o "${temporary_dir}/${source}-sanitized"
  ASAN_OPTIONS=detect_leaks="${I2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${temporary_dir}/${source}-sanitized" >/dev/null
done

if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/include/eshkol_transformer/f32_tensor.h"; then
  die "I2 production path contains a forbidden Python/PyTorch reference"
fi

printf 'I2 PASS: C/C++ ABI, exact-f32 storage, gradients, transactions, K1 views, determinism, and sanitizers\n'
