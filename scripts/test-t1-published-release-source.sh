#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-t1-release.XXXXXX")"
trap 'rm -rf -- "${tmp}"' EXIT
cc="${CC:-cc}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fno-common
       -DET_T1_PUBLISHED_RELEASE_PRIVATE -DET_T1_I64_SHELL_TESTING
       -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE -DET_I64_TENSOR_TESTING
       -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
sources=("${PROJECT_ROOT}/tests/t1/published_release_native.c"
         "${PROJECT_ROOT}/native/t1_i64_shell.c"
         "${PROJECT_ROOT}/native/i64_tensor.c"
         "${PROJECT_ROOT}/native/kernel_abi.c")
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.t1.test_published_release_source
"${cc}" "${flags[@]}" -O2 "${sources[@]}" -o "${tmp}/release"
"${tmp}/release" | tee "${tmp}/normal.stdout"
"${cc}" "${flags[@]}" -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined "${sources[@]}" -o "${tmp}/sanitize"
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${tmp}/sanitize" | tee "${tmp}/sanitize.stdout"
cmp "${tmp}/normal.stdout" "${tmp}/sanitize.stdout"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/t1_i64_shell.c" -o "${tmp}/feature-off.o"
if nm -g --defined-only "${tmp}/feature-off.o" | \
   grep -E 'et_t1_i64_shell_private_release(_preflight)?_v1'; then
  exit 1
fi
