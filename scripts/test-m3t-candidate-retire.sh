#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
ulimit -c 0
cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
out="$(mktemp -d)"
trap 'rm -rf -- "$out"' EXIT
base=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all
  -DET_G3R_CANDIDATE_RETIRE_PRIVATE -DET_F32_TENSOR_TESTING
  -DET_I64_TENSOR_TESTING -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
sources=("${PROJECT_ROOT}/tests/m3t/test_candidate_retire.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/i64_tensor.c" "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/n2_primitives_provider.c"
  "${PROJECT_ROOT}/native/a2_attention_provider.c")
for mode in normal repeat sanitize; do
  flags=(-O2)
  if [[ "$mode" == sanitize ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${base[@]}" "${flags[@]}" "${sources[@]}" -lm -o "$out/$mode"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 "$out/$mode"
done
