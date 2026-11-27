#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
cc="${CC:-clang}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all
  -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING -DET_PA4_WIDER_M3T_DIAGNOSTIC
  -I include -I native)
sources=(tests/m3t/test_transport.c src/eshkol_transformer/m3t_f32_integration.c
  tests/m3t/kernel_fail_allocator.c native/i64_tensor.c native/t1_i64_shell.c
  native/n3k_primitives_provider.c native/n2_primitives_provider.c
  native/a2_attention_provider.c native/pa4_wider_attention_provider.c)
for mode in normal sanitize; do
  optimization=(-O2)
  if [[ "${mode}" == sanitize ]]; then
    optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${cc}" "${flags[@]}" "${optimization[@]}" "${sources[@]}" \
    -Wl,--wrap=et_a2_kernel_provider_v1 -lm -o "${scratch}/diagnostic"
  for provider in a2 reference; do
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      "${scratch}/diagnostic" "${provider}" >"${scratch}/${provider}.bin"
  done
  # Two cases, each with 512 logits, four attention outputs of eight f32
  # elements each, and 1,184 elements across the 14 parameter VJPs.
  test "$(wc -c <"${scratch}/a2.bin")" -eq "$((2 * (512 + 4 * 8 + 1184) * 4))"
  cmp "${scratch}/a2.bin" "${scratch}/reference.bin"
  printf 'PA4-Wider M3T %s: 3,456 f32 values bitwise equal, forward/reverse rejection and lifetime PASS\n' "${mode}"
done
