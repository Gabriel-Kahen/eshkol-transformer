#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -fno-fast-math -I include)
sources=(tests/paged_kv/test_mod5_gqa.c native/mod5_primitives_provider.c
         native/paged_kv.c native/kernel_abi.c)

"$cc" "${flags[@]}" -O2 "${sources[@]}" -lm -o "$scratch/witness"
"$scratch/witness" "$scratch/normal.bits" >"$scratch/normal.out"
"$scratch/witness" "$scratch/repeat.bits" >"$scratch/repeat.out"
"$cc" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${sources[@]}" -lm -o "$scratch/witness.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$scratch/witness.san" "$scratch/san.bits" >"$scratch/san.out"
for extension in out bits; do
  cmp "$scratch/normal.$extension" "$scratch/repeat.$extension"
  cmp "$scratch/normal.$extension" "$scratch/san.$extension"
done
cat "$scratch/normal.out"
printf 'MOD5/PKV5 native normal/repeat/ASan/UBSan/LSan parity: PASS\n'
