#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in "${CC:-clang}" "${CXX:-clang++}" nm python3; do
  require_command "${command}"
done

scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -frounding-math -fno-fast-math
  -fstack-protector-all -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src/eshkol_transformer")
dependencies=(
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_call_f32_integration.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_i64_integration.c"
  "${PROJECT_ROOT}/native/kernel_abi.c"
  "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/n2_primitives_provider.c"
  "${PROJECT_ROOT}/native/a2_attention_provider.c"
  "${PROJECT_ROOT}/native/indexed_cross_entropy.c"
  "${PROJECT_ROOT}/native/l3s_masked_objective_provider.c"
  "${PROJECT_ROOT}/native/e3_evaluation_metrics_provider.c")

"${CXX:-clang++}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -DET_E3_R1_PREFLIGHT_TESTING -I "${PROJECT_ROOT}/include" \
  -I "${PROJECT_ROOT}/src/eshkol_transformer" \
  -c "${PROJECT_ROOT}/tests/e3_native/header_cpp.cpp" \
  -o "${scratch}/header.o"

for mode in normal sanitize; do
  extra=(-O2)
  if [[ "${mode}" == sanitize ]]; then
    extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
    "${PROJECT_ROOT}/tests/e3_native/test_r1_preflight.c" \
    "${dependencies[@]}" -lm -o "${scratch}/r1-${mode}"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "${scratch}/r1-${mode}"
done

"${CC:-clang}" "${flags[@]}" -O2 \
  "${PROJECT_ROOT}/tests/e3_native/test_frame.c" \
  "${dependencies[@]}" -lm -o "${scratch}/base"
"${scratch}/base"
"${PYTHON:-python3}" -m unittest \
  tests.e3_private.test_r1_preflight_source \
  tests.e3_private.test_r1_cut_inventory -v
