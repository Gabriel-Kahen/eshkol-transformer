#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
m3t_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
m3t_test_dir="$(mktemp -d)"
trap 'rm -rf -- "${m3t_test_dir}"' EXIT
m3t_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all
  -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
m3t_sources=("${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c"
  "${PROJECT_ROOT}/native/kernel_abi.c")
for m3t_mode in normal sanitize; do
  m3t_mode_flags=(-O2)
  if [[ "$m3t_mode" == sanitize ]]; then
    m3t_mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$m3t_cc" "${m3t_flags[@]}" "${m3t_mode_flags[@]}" \
    "${PROJECT_ROOT}/tests/m3t/test_scoped.c" "${m3t_sources[@]}" \
    -lm -o "${m3t_test_dir}/scoped-${m3t_mode}"
  "$m3t_cc" "${m3t_flags[@]}" "${m3t_mode_flags[@]}" \
    "${PROJECT_ROOT}/tests/m3t/test_transport.c" \
    "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" \
    "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c" \
    "${PROJECT_ROOT}/native/i64_tensor.c" "${PROJECT_ROOT}/native/t1_i64_shell.c" \
    "${PROJECT_ROOT}/native/n3k_primitives_provider.c" \
    "${PROJECT_ROOT}/native/n2_primitives_provider.c" \
    "${PROJECT_ROOT}/native/a2_attention_provider.c" \
    -lm -o "${m3t_test_dir}/transport-${m3t_mode}"
  ASAN_OPTIONS="detect_leaks=${M3T_ASAN_DETECT_LEAKS:-1}:halt_on_error=1" \
    UBSAN_OPTIONS=halt_on_error=1 "${m3t_test_dir}/scoped-${m3t_mode}"
  ASAN_OPTIONS="detect_leaks=${M3T_ASAN_DETECT_LEAKS:-1}:halt_on_error=1" \
    UBSAN_OPTIONS=halt_on_error=1 "${m3t_test_dir}/transport-${m3t_mode}"
done
