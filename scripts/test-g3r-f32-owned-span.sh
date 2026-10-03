#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
baseline=acbf623762d9453ab88da10f47f55669340f2279
if [[ $# -ne 2 || "$1" != /* || "$2" != /* || ! -x "$1" || -e "$2" || -L "$2" ]]; then
  echo 'usage: test-g3r-f32-owned-span.sh /absolute/compiler /fresh/absolute/output' >&2
  exit 2
fi
cc="$1"
out="$2"
mkdir -p -- "$out/base" "$out/off" "$out/on" "$out/normal" \
  "$out/repeat" "$out/san"
git -C "$root" show "$baseline:native/f32_tensor.c" >"$out/base/f32_tensor.c"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -I "$root/include" -I "$root/native")
"$cc" "${flags[@]}" -c "$out/base/f32_tensor.c" -o "$out/base/off.o" \
  >"$out/base/off.stdout" 2>"$out/base/off.stderr"
"$cc" "${flags[@]}" -c "$root/native/f32_tensor.c" -o "$out/off/f32.o" \
  >"$out/off/compile.stdout" 2>"$out/off/compile.stderr"
cmp "$out/base/off.o" "$out/off/f32.o"
[[ ! -s "$out/base/off.stderr" && ! -s "$out/off/compile.stderr" ]]
nm -g --defined-only --format=posix "$out/off/f32.o" | cut -d' ' -f1 | \
  sort >"$out/off/names"
! grep -F 'et_f32_tensor_private_owned_span_overlap_v1' "$out/off/names"

"$cc" "${flags[@]}" -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -c "$out/base/f32_tensor.c" -o "$out/base/on.o" \
  >"$out/base/on.stdout" 2>"$out/base/on.stderr"
"$cc" "${flags[@]}" -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -c "$root/native/f32_tensor.c" -o "$out/on/f32.o" \
  >"$out/on/compile.stdout" 2>"$out/on/compile.stderr"
[[ ! -s "$out/base/on.stderr" && ! -s "$out/on/compile.stderr" ]]
nm -g --defined-only --format=posix "$out/base/on.o" | cut -d' ' -f1 | \
  sort >"$out/base/on.names"
nm -g --defined-only --format=posix "$out/on/f32.o" | cut -d' ' -f1 | \
  sort >"$out/on/names"
comm -13 "$out/base/on.names" "$out/on/names" >"$out/on/delta"
printf '%s\n' et_f32_tensor_private_owned_span_overlap_v1 \
  >"$out/on/expected.delta"
cmp "$out/on/expected.delta" "$out/on/delta"

for mode in normal repeat san; do
  mode_flags=("${flags[@]}")
  if [[ "$mode" == san ]]; then
    mode_flags+=(-g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined,leak)
  fi
  "$cc" "${mode_flags[@]}" "$root/native/kernel_abi.c" \
    "$root/tests/i2/test_g3r_f32_owned_span.c" -o "$out/$mode/test" \
    >"$out/$mode/compile.stdout" 2>"$out/$mode/compile.stderr"
  [[ ! -s "$out/$mode/compile.stderr" ]]
  set +e
  ASAN_OPTIONS=detect_leaks=1 timeout --foreground --signal=TERM --kill-after=5s \
    90s "$out/$mode/test" >"$out/$mode/run.stdout" 2>"$out/$mode/run.stderr"
  status=$?
  set -e
  printf '%s\n' "$status" >"$out/$mode/run.exit"
  [[ "$status" -eq 0 && ! -s "$out/$mode/run.stderr" ]]
  grep -E '^G3R f32 owned span PASS: [0-9]+ checks$' \
    "$out/$mode/run.stdout" >/dev/null
done
cmp "$out/normal/run.stdout" "$out/repeat/run.stdout"
cmp "$out/normal/run.stdout" "$out/san/run.stdout"
printf 'G3R f32 owned span gate PASS\n'
