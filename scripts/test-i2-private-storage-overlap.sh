#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evidence="${1:-${project_root}/build/i2-private-storage-overlap}"
cc="${CC:-clang}"
mkdir -p "${evidence}"
for command in "${cc}" cmp comm nm sha256sum; do
  command -v "${command}" >/dev/null 2>&1 || {
    printf 'missing command: %s\n' "${command}" >&2
    exit 1
  }
done

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
       -ffp-contract=off -fexcess-precision=standard -fno-fast-math
       -fvisibility=hidden -I "${project_root}/include"
       -I "${project_root}/native" -I "${project_root}/src"
       -I "${project_root}/src/eshkol_transformer")
"${cc}" "${flags[@]}" -O2 -c "${project_root}/native/f32_tensor.c" \
  -o "${evidence}/f32-feature-off.o"
"${cc}" "${flags[@]}" -O2 -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
  -c "${project_root}/native/f32_tensor.c" \
  -o "${evidence}/f32-private.o"
for variant in feature-off private; do
  nm -g --defined-only --format=posix "${evidence}/f32-${variant}.o" |
    awk 'NF >= 2 {print $1}' | LC_ALL=C sort -u \
    >"${evidence}/f32-${variant}-defined.txt"
  nm -u --format=posix "${evidence}/f32-${variant}.o" |
    awk 'NF >= 1 {print $1}' | LC_ALL=C sort -u \
    >"${evidence}/f32-${variant}-undefined.txt"
done
comm -13 "${evidence}/f32-feature-off-defined.txt" \
  "${evidence}/f32-private-defined.txt" >"${evidence}/added-defined.txt"
printf '%s\n' et_f32_tensor_private_storage_overlap_v1 |
  cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/f32-feature-off-undefined.txt" \
  "${evidence}/f32-private-undefined.txt" \
  >"${evidence}/added-undefined.txt"
test ! -s "${evidence}/added-undefined.txt"

run_mode() {
  local mode="$1"
  local -a mode_flags=(-O2) environment=()
  if [[ "${mode}" == sanitizer ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "${cc}" "${flags[@]}" "${mode_flags[@]}" \
    "${project_root}/tests/i2/test_alias_envelope.c" \
    "${project_root}/native/kernel_abi.c" -lm \
    -o "${evidence}/i2-alias-${mode}"
  for case_name in coverage product endpoint uninitialized; do
    "${environment[@]}" "${evidence}/i2-alias-${mode}" "${case_name}" \
      >"${evidence}/i2-${mode}-${case_name}.stdout" \
      2>"${evidence}/i2-${mode}-${case_name}.stderr"
    test ! -s "${evidence}/i2-${mode}-${case_name}.stderr"
  done
  "${cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_G3C4_GENERATOR_A2_TU \
    -c "${project_root}/tests/g3c4/test_generator_native.c" \
    -o "${evidence}/g3-a2-${mode}.o"
  "${cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE \
    "${project_root}/tests/g3c4/test_generator_native.c" \
    "${project_root}/tests/m3t/kernel_fail_allocator.c" \
    "${project_root}/native/n3k_primitives_provider.c" \
    "${evidence}/g3-a2-${mode}.o" -lm \
    -o "${evidence}/g3-closed-generator-${mode}"
  "${environment[@]}" ET_I2_PRIVATE_OVERLAP_FOCUSED=1 \
    "${evidence}/g3-closed-generator-${mode}" \
    >"${evidence}/g3-${mode}.stdout" \
    2>"${evidence}/g3-${mode}.stderr"
  test ! -s "${evidence}/g3-${mode}.stderr"
}

run_mode normal
ET_I2_PRIVATE_OVERLAP_FOCUSED=1 "${evidence}/g3-closed-generator-normal" \
  >"${evidence}/g3-repeat.stdout" 2>"${evidence}/g3-repeat.stderr"
test ! -s "${evidence}/g3-repeat.stderr"
cmp "${evidence}/g3-normal.stdout" "${evidence}/g3-repeat.stdout"
run_mode sanitizer
cmp "${evidence}/g3-normal.stdout" "${evidence}/g3-sanitizer.stdout"
for case_name in coverage product endpoint uninitialized; do
  cmp "${evidence}/i2-normal-${case_name}.stdout" \
      "${evidence}/i2-sanitizer-${case_name}.stdout"
done
sha256sum "${project_root}/native/f32_tensor.c" \
  "${project_root}/include/eshkol_transformer/f32_tensor.h" \
  "${project_root}/tests/i2/test_alias_envelope.c" \
  "${project_root}/tests/g3c4/test_generator_native.c" \
  >"${evidence}/source.sha256"
printf 'I2 private storage-overlap PASS: %s\n' "${evidence}"
