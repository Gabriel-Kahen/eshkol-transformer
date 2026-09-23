#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_supported_host

e3_cc="${CC:-clang}"
e3_cxx="${CXX:-clang++}"
require_command "${e3_cc}"
require_command "${e3_cxx}"
e3_expected="$(lock_value clang_version)"
e3_cc_version="$("${e3_cc}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
e3_cxx_version="$("${e3_cxx}" --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
check_supported_version Clang "${e3_cc_version}" "${e3_expected}"
check_supported_version Clang++ "${e3_cxx_version}" "${e3_expected}"

e3_tmp="$(mktemp -d)"
trap 'rm -rf -- "${e3_tmp}"' EXIT
e3_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -frounding-math -fno-fast-math
  -fstack-protector-all -DET_F32_TENSOR_TESTING -DET_I64_TENSOR_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src/eshkol_transformer")
e3_sources=("${PROJECT_ROOT}/tests/e3_native/test_frame.c"
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

"${e3_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/src/eshkol_transformer" \
  "${PROJECT_ROOT}/tests/e3_native/header_cpp.cpp" -c \
  -o "${e3_tmp}/header_cpp.o"

for e3_mode in normal sanitize; do
  e3_extra=(-O2)
  if [[ "${e3_mode}" == sanitize ]]; then
    e3_extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${e3_cc}" "${e3_flags[@]}" "${e3_extra[@]}" "${e3_sources[@]}" \
    -lm -o "${e3_tmp}/frame-${e3_mode}"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "${e3_tmp}/frame-${e3_mode}"
done
