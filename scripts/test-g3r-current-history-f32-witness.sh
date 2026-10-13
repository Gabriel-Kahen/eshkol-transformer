#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/g3r-current-f32-witness.XXXXXX")
trap 'rm -rf -- "$out"' EXIT
for mode in normal repeat sanitizer; do
  flags=(-O0)
  if [[ "$mode" == sanitizer ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  cc -std=c11 -Wall -Wextra -Werror "${flags[@]}" \
    -I"$root/include" -I"$root/native" -I"$root/src" \
    "$root/tests/g3r/current_history_f32_witness_test.c" \
    "$root/native/kernel_abi.c" -lm -o "$out/$mode"
  if [[ "$mode" == sanitizer ]]; then
    ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 "$out/$mode"
  else
    "$out/$mode"
  fi
done
cc -std=c11 -O0 -Wall -Wextra -Werror -I"$root/include" -I"$root/native" \
  -c "$root/native/f32_tensor.c" -o "$out/feature-off.o"
if nm -g --defined-only "$out/feature-off.o" | \
    rg -q '^.*et_f32_parameter_save_witness_(begin|check|end)_v1$'; then
  echo "F32 witness escaped feature-off object" >&2
  exit 1
fi
echo 'F32 witness feature-off symbols PASS'
