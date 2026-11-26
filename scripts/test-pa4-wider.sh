#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
out="${root}/build/pa4_wider"
archive="${out}/libeshkol_transformer_pa4_wider.a"
sources=(native/pa4_wider_attention_provider.c native/kernel_abi.c)
common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -I include)
flags=(-std=c11 "${common[@]}")

CC="${cc}" bash scripts/build-pa4-wider.sh
"${cc}" "${flags[@]}" -O2 tests/pa4_wider/test_pa4_wider_provider.c \
  "${archive}" native/kernel_abi.c -lm -o "${out}/test-provider"
"${out}/test-provider"
"${cxx}" -std=c++17 "${common[@]}" tests/pa4_wider/test_pa4_wider_header_cpp.cpp \
  "${archive}" -lm -o "${out}/test-header"
"${out}/test-header"
printf 'PA4-Wider C++17 ABI: PASS\n'

"${cc}" "${flags[@]}" -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
  -fno-sanitize-recover=all tests/pa4_wider/test_pa4_wider_provider.c \
  "${sources[@]}" -lm -o "${out}/test-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${out}/test-sanitized"
printf 'PA4-Wider ASan/UBSan/LSan: PASS\n'

"${cc}" "${flags[@]}" -O2 tests/pa4_wider/bench_pa4_wider.c \
  "${archive}" native/kernel_abi.c -lm -o "${out}/bench-pa4-wider"
"${out}/bench-pa4-wider"
