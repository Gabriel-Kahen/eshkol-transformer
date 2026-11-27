#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
bash scripts/build-mod5.sh
bash scripts/build-dist5.sh

scratch="$(mktemp -d "${root}/build/dist5/.weighted-test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fno-fast-math -fexcess-precision=standard -I include)
test_source=tests/dist5/test_dist5_weighted_sgd.c
"${cc}" "${flags[@]}" -O2 -c native/kernel_abi.c -o "${scratch}/kernel.o"
"${cc}" "${flags[@]}" -O2 "${test_source}" \
  build/mod5/libeshkol_transformer_mod5.a "${scratch}/kernel.o" \
  build/dist5/libeshkol_dist5.a -lm -o "${scratch}/test"
"${scratch}/test"

export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
"${cc}" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${test_source}" native/mod5_primitives_provider.c native/kernel_abi.c \
  native/dist5_collectives.c -lm -o "${scratch}/test-sanitized"
"${scratch}/test-sanitized"
printf 'DIST5 weighted SGD ASan/UBSan/LSan: PASS\n'
