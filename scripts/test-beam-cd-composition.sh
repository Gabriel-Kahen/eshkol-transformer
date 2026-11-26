#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
cc="${CC:-clang}"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include)
src=(tests/beam_cd/test_beam_cd_composition.c native/beam_search.c
     native/constrained_decode.c)

"${cc}" "${common[@]}" -O2 "${src[@]}" -lm -o "${scratch}/witness"
"${scratch}/witness"
"${cc}" "${common[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -fno-sanitize-recover=all "${src[@]}" -lm \
  -o "${scratch}/witness-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "${scratch}/witness-san"
