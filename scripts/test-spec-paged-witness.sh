#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
cc="${CC:-clang}"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include)
source=(tests/spec_paged/test_spec_paged_witness.c
        native/spec_decode.c native/paged_kv.c)

"${cc}" "${common[@]}" -O2 "${source[@]}" -lm -o "${scratch}/witness"
"${scratch}/witness" > "${scratch}/normal.out"
"${cc}" "${common[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${source[@]}" -lm -o "${scratch}/witness.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/witness.san" > "${scratch}/san.out"
cmp "${scratch}/normal.out" "${scratch}/san.out"
cat "${scratch}/normal.out"
printf 'SPEC_PAGED normal/ASan/UBSan/LSan parity: PASS\n'
