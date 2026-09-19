#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
m3_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
m3_tmp="$(mktemp -d)"
trap 'rm -rf -- "${m3_tmp}"' EXIT
m3_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -frounding-math -fno-fast-math -fstack-protector-all
  -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING -DET_M3_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
m3_sources=("${PROJECT_ROOT}/tests/m3/test_model_native.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_i64_integration.c"
  "${PROJECT_ROOT}/native/kernel_abi.c" "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/n2_primitives_provider.c"
  "${PROJECT_ROOT}/native/a2_attention_provider.c")
for m3_mode in normal sanitize; do
  m3_extra=(-O2); m3_args=()
  if [[ "$m3_mode" == sanitize ]]; then
    m3_extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    # Full failure/ownership cases in both modes; quadratic retained-registry
    # scans make the sanitizer trajectory bounded at 100, normal at 1000.
    m3_args=(100)
  fi
  "$m3_cc" "${m3_flags[@]}" "${m3_extra[@]}" "${m3_sources[@]}" -lm \
    -o "${m3_tmp}/native-${m3_mode}"
  "$m3_cc" "${m3_flags[@]}" "${m3_extra[@]}" \
    "${PROJECT_ROOT}/tests/probes/m3_lifetime/plan_retention.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -o "${m3_tmp}/plans-${m3_mode}"
  ASAN_OPTIONS="detect_leaks=${M3_ASAN_DETECT_LEAKS:-1}:halt_on_error=1" \
    UBSAN_OPTIONS=halt_on_error=1 "${m3_tmp}/plans-${m3_mode}"
  TIMEFORMAT="M3 NATIVE ${m3_mode} elapsed=%R seconds"
  time ASAN_OPTIONS="detect_leaks=${M3_ASAN_DETECT_LEAKS:-1}:halt_on_error=1" \
    UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
    "${m3_tmp}/native-${m3_mode}" "${m3_args[@]}"
done
