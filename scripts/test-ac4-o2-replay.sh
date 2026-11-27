#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -fno-fast-math -frounding-math
       -Iinclude -Inative -DET_O2_TESTING)
sources=(tests/ac4/test_ac4_o2_replay.c native/ac4_checkpointing.c
         native/mod5_primitives_provider.c native/kernel_abi.c
         native/f32_tensor.c native/o2_optimizer.c)
"$cc" "${flags[@]}" -O2 "${sources[@]}" -lm -o "$scratch/native"
"$scratch/native" "$scratch/native.bits" >"$scratch/native.stdout"
"$scratch/native" "$scratch/repeat.bits" >"$scratch/repeat.stdout"
"$cc" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${sources[@]}" -lm -o "$scratch/sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$scratch/sanitized" "$scratch/sanitized.bits" >"$scratch/sanitized.stdout"
for extension in bits stdout; do
  cmp "$scratch/native.$extension" "$scratch/repeat.$extension"
  cmp "$scratch/native.$extension" "$scratch/sanitized.$extension"
done
cat "$scratch/native.stdout"
printf 'AC4/O2 normal/repeat/ASan/UBSan/LSan trace parity: PASS\n'
