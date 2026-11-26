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
