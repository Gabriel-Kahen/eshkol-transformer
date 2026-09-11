#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp grep timeout; do
  require_command "${command}"
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
artifact_dir="$(project_build_dir)/i2"
library="${artifact_dir}/libeshkol_transformer_f32.a"
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-i2-native.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

[[ -r "${library}" && -r "${k1_library}" ]] || \
  die "I2 native or K1 archive is missing"
[[ "$(ar t "${library}")" == "f32_tensor.o" ]] || \
  die "I2 native archive has unexpected members"

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)

for source in test_f32_tensor test_f32_parameter; do
  "${cc}" "${cflags[@]}" "${PROJECT_ROOT}/tests/i2/${source}.c" \
    "${library}" "${k1_library}" -o "${temporary_dir}/${source}"
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 60s \
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

"${PROJECT_ROOT}/scripts/build-k1.sh" "${temporary_dir}/sanitized-k1" sanitize
"${PROJECT_ROOT}/scripts/build-i2.sh" "${temporary_dir}/sanitized-i2" sanitize-test
for source in test_f32_tensor test_f32_parameter; do
  "${cc}" "${cflags[@]}" -DET_F32_TENSOR_TESTING \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    "${PROJECT_ROOT}/tests/i2/${source}.c" \
    "${temporary_dir}/sanitized-i2/libeshkol_transformer_f32.a" \
    "${temporary_dir}/sanitized-k1/libeshkol_transformer_k1.a" \
    -o "${temporary_dir}/${source}-sanitized"
  ASAN_OPTIONS=detect_leaks="${I2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s \
      "${temporary_dir}/${source}-sanitized" >/dev/null
done

if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/include/eshkol_transformer/f32_tensor.h"; then
  die "I2 native path contains a forbidden Python/PyTorch reference"
fi

printf 'I2 NATIVE PASS: storage, gradients, transactions, C/C++ ABI, determinism, and sanitizers\n'
