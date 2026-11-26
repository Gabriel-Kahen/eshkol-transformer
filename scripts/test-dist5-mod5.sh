#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
bash scripts/build-mod5.sh
bash scripts/build-dist5.sh

scratch="$(mktemp -d "${root}/build/dist5/.mod5-test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fno-fast-math -fexcess-precision=standard -I include)
"${cc}" "${flags[@]}" -O2 -c native/kernel_abi.c -o "${scratch}/kernel.o"
"${cc}" "${flags[@]}" -O2 tests/dist5/test_dist5_mod5_gradients.c \
  build/mod5/libeshkol_transformer_mod5.a "${scratch}/kernel.o" \
  build/dist5/libeshkol_dist5.a -lm -o "${scratch}/test"
"${scratch}/test"

sanitizers=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
for source in mod5_primitives_provider kernel_abi dist5_collectives; do
  "${cc}" "${flags[@]}" "${sanitizers[@]}" -c "native/${source}.c" \
    -o "${scratch}/${source}.san.o"
done
"${cc}" "${flags[@]}" "${sanitizers[@]}" \
  tests/dist5/test_dist5_mod5_gradients.c \
  "${scratch}/mod5_primitives_provider.san.o" \
  "${scratch}/kernel_abi.san.o" "${scratch}/dist5_collectives.san.o" \
  -lm -o "${scratch}/test-sanitized"
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
"${scratch}/test-sanitized"
printf 'DIST5/MOD5 ASan/UBSan/LSan: PASS\n'
