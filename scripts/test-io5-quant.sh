#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${root}/build/io5_quant"
bash scripts/build-io5-quant.sh

scratch="$(mktemp -d "${artifact}/.test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

cp "${artifact}/libeshkol_transformer_io5_quant.a" "${scratch}/first.a"
cp "${artifact}/libeshkol_transformer_io5_quant.so" "${scratch}/first.so"
bash scripts/build-io5-quant.sh
cmp "${scratch}/first.a" "${artifact}/libeshkol_transformer_io5_quant.a"
cmp "${scratch}/first.so" "${artifact}/libeshkol_transformer_io5_quant.so"
printf 'IO5 quant artifact reproducibility (.a and .so): PASS\n'

common=(-Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math -I include)
cflags=(-std=c11 "${common[@]}" -fexcess-precision=standard)
cppflags=(-std=c++17 "${common[@]}")
sanitizers=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free)

# 1. C++17 header and linkage check.
"${CXX}" "${cppflags[@]}" tests/io5/header_quant_cpp.cpp \
  "${artifact}/libeshkol_transformer_io5_quant.a" \
  -o "${scratch}/header-quant-cpp"
"${scratch}/header-quant-cpp"
printf 'IO5 quant C++17 header and linkage: PASS\n'

# 2. Normal C11 native tests with allocation-free guard.
"${CC}" "${cflags[@]}" -O2 -DET_IO5_QUANT_WRAP_ALLOC \
  tests/io5/test_io5_quant_native.c \
  "${artifact}/libeshkol_transformer_io5_quant.a" \
  "${wrap[@]}" -lm -o "${scratch}/quant-native-normal"
"${scratch}/quant-native-normal"

# 3. ASan/UBSan/LSan C11 native tests.
"${CC}" "${cflags[@]}" "${sanitizers[@]}" \
  tests/io5/test_io5_quant_native.c \
  native/io5_quant_codec.c \
  -lm -o "${scratch}/quant-native-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/quant-native-san"
printf 'IO5 quant ASan/UBSan/LSan: PASS\n'

# 4. Python quantized interchange and tamper/negative tests via native C11 .so.
IO5_QUANT_REQUIRED=1 ESHKOL_IO5_QUANT_LIB="${artifact}/libeshkol_transformer_io5_quant.so" \
  python3 -m unittest -v tests/io5/test_quantized_interchange.py
printf 'IO5 quantized interchange suite: PASS\n'
