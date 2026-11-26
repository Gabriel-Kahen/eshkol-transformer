#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
bash scripts/build-mod5.sh
bash scripts/build-mp4.sh

scratch="$(mktemp -d "${root}/build/mp4/.mod5-test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fno-fast-math -fexcess-precision=standard -I include)
"${cc}" "${flags[@]}" -O2 -c native/kernel_abi.c -o "${scratch}/kernel.o"
"${cc}" "${flags[@]}" -O2 tests/mp4/test_mp4_mod5_gradients.c \
  build/mod5/libeshkol_transformer_mod5.a \
  build/mp4/libeshkol_transformer_mp4.a "${scratch}/kernel.o" \
  -lm -o "${scratch}/test"
"${scratch}/test"

sanitizers=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${cc}" "${flags[@]}" "${sanitizers[@]}" \
  tests/mp4/test_mp4_mod5_gradients.c native/mod5_primitives_provider.c \
  native/mp4_mixed_precision_provider.c native/kernel_abi.c \
  -lm -o "${scratch}/test-sanitized"
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
"${scratch}/test-sanitized"
printf 'MP4/MOD5 ASan/UBSan/LSan: PASS\n'
