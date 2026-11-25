#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
bash scripts/build-dist5.sh

scratch="$(mktemp -d "${root}/build/dist5/.test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fno-fast-math -fexcess-precision=standard -I include)
sources=(tests/dist5/test_dist5_collectives.c)
"${cc}" "${flags[@]}" -O2 "${sources[@]}" build/dist5/libeshkol_dist5.a -lm \
  -o "${scratch}/test"
"${scratch}/test"

lsan="${DIST5_ASAN_DETECT_LEAKS:-0}"
export ASAN_OPTIONS="detect_leaks=${lsan}:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
"${cc}" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${sources[@]}" native/dist5_collectives.c -lm -o "${scratch}/test-sanitized"
"${scratch}/test-sanitized"
printf 'DIST5 ASan/UBSan: PASS (LSan detect_leaks=%s)\n' "${lsan}"
