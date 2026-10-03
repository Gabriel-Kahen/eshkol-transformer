#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
baseline=10090ac4b158940fa7beba8efddf07c71d7c2a3d
if [[ $# -ne 2 || "$1" != /* || "$2" != /* || ! -x "$1" || -e "$2" || -L "$2" ]]; then
  echo 'usage: test-g3r-f32-destroy-ready.sh /absolute/compiler /fresh/absolute/output' >&2
  exit 2
fi
cc="$1"
out="$2"
mkdir -p -- "$out/base" "$out/candidate" "$out/normal" "$out/repeat" "$out/san"
[[ "$(git -C "$root" rev-parse "$baseline")" == "$baseline" ]]
git -C "$root" show "$baseline:native/f32_tensor.c" >"$out/base/f32_tensor.c"
git -C "$root" show "$baseline:native/f32_parameter_internal.h" >"$out/base/f32_parameter_internal.h"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -I "$root/include" -I "$root/native")
"$cc" "${flags[@]}" -c "$out/base/f32_tensor.c" -o "$out/base/f32_tensor.o" \
  >"$out/base/compile.stdout" 2>"$out/base/compile.stderr"
"$cc" "${flags[@]}" -c "$root/native/f32_tensor.c" \
  -o "$out/candidate/f32_tensor.o" \
  >"$out/candidate/compile.stdout" 2>"$out/candidate/compile.stderr"
cmp "$out/base/f32_tensor.o" "$out/candidate/f32_tensor.o"
[[ ! -s "$out/base/compile.stderr" && ! -s "$out/candidate/compile.stderr" ]]
nm -g --defined-only --format=posix "$out/candidate/f32_tensor.o" \
  >"$out/candidate/off.symbols"
! grep -F 'et_f32_tensor_private_destroy_ready_v1 ' "$out/candidate/off.symbols"

for mode in normal repeat san; do
  mode_flags=("${flags[@]}" -DET_F32_TENSOR_TESTING
    -DET_G3R_CANDIDATE_RETIRE_PRIVATE)
  if [[ "$mode" == san ]]; then
    mode_flags+=(-g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined,leak)
  fi
  "$cc" "${mode_flags[@]}" "$root/native/f32_tensor.c" \
    "$root/native/kernel_abi.c" "$root/tests/i2/test_g3r_f32_destroy_ready.c" \
    -o "$out/$mode/test" >"$out/$mode/compile.stdout" \
    2>"$out/$mode/compile.stderr"
  [[ ! -s "$out/$mode/compile.stderr" ]]
  nm -g --defined-only --format=posix "$out/$mode/test" \
    >"$out/$mode/on.symbols"
  grep -F 'et_f32_tensor_private_destroy_ready_v1 ' "$out/$mode/on.symbols" >/dev/null
  set +e
  ASAN_OPTIONS=detect_leaks=1 timeout --foreground --signal=TERM --kill-after=5s \
    90s "$out/$mode/test" >"$out/$mode/run.stdout" 2>"$out/$mode/run.stderr"
  status=$?
  set -e
  printf '%s\n' "$status" >"$out/$mode/run.exit"
  [[ "$status" -eq 0 ]]
  [[ ! -s "$out/$mode/run.stderr" ]]
  grep -E '^G3R f32 destroy readiness PASS: [0-9]+ checks$' \
    "$out/$mode/run.stdout" >/dev/null
done
cmp "$out/normal/run.stdout" "$out/repeat/run.stdout"
cmp "$out/normal/run.stdout" "$out/san/run.stdout"
printf 'G3R f32 destroy readiness gate PASS\n'
