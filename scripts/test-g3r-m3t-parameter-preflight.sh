#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
baseline=779fddd47b5c23dd8afa20cdecf4024318a3a533
if [[ $# -ne 2 || "$1" != /* || "$2" != /* || ! -x "$1" || -e "$2" || -L "$2" ]]; then
  echo 'usage: test-g3r-m3t-parameter-preflight.sh /absolute/compiler /fresh/absolute/output' >&2
  exit 2
fi
cc="$1"
out="$2"
mkdir -p -- "$out/base/src/eshkol_transformer" "$out/off" "$out/on" \
  "$out/normal" "$out/repeat" "$out/san"
[[ "$(git -C "$root" rev-parse "$baseline")" == "$baseline" ]]
git -C "$root" show "$baseline:src/eshkol_transformer/m3t_transport.c" \
  >"$out/base/src/eshkol_transformer/m3t_transport.c"
ln -s "$root/native" "$out/base/native"
ln -s "$root/src/eshkol_transformer/m3t_transport.h" \
  "$out/base/src/eshkol_transformer/m3t_transport.h"
ln -s "$root/src/eshkol_transformer/m3t_f32_scoped.h" \
  "$out/base/src/eshkol_transformer/m3t_f32_scoped.h"

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -I "$root/include" -I "$root/native")
"$cc" "${flags[@]}" -O2 -c \
  "$out/base/src/eshkol_transformer/m3t_transport.c" -o "$out/base/transport.o" \
  >"$out/base/compile.stdout" 2>"$out/base/compile.stderr"
"$cc" "${flags[@]}" -O2 -c "$root/src/eshkol_transformer/m3t_transport.c" \
  -o "$out/off/transport.o" >"$out/off/compile.stdout" \
  2>"$out/off/compile.stderr"
cmp "$out/base/transport.o" "$out/off/transport.o"
[[ ! -s "$out/base/compile.stderr" && ! -s "$out/off/compile.stderr" ]]
nm -g --defined-only --format=posix "$out/off/transport.o" >"$out/off/symbols"
! grep -F 'et_m3t_private_candidate_retire_preflight_v1 ' "$out/off/symbols"
"$cc" "${flags[@]}" -O2 -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -c "$root/src/eshkol_transformer/m3t_transport.c" \
  -o "$out/on/transport.o" >"$out/on/compile.stdout" \
  2>"$out/on/compile.stderr"
[[ ! -s "$out/on/compile.stderr" ]]
nm -g --defined-only --format=posix "$out/on/transport.o" >"$out/on/symbols"
grep -F 'et_m3t_private_candidate_retire_preflight_v1 ' "$out/on/symbols" >/dev/null

sources=("$root/tests/m3t/test_g3r_candidate_preflight.c"
  "$root/src/eshkol_transformer/m3t_f32_integration.c"
  "$root/native/i64_tensor.c" "$root/native/t1_i64_shell.c"
  "$root/native/n3k_primitives_provider.c"
  "$root/native/n2_primitives_provider.c"
  "$root/native/a2_attention_provider.c" "$root/native/kernel_abi.c")
for mode in normal repeat san; do
  mode_flags=(-O2)
  if [[ "$mode" == san ]]; then
    mode_flags=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined)
  fi
  "$cc" "${flags[@]}" "${mode_flags[@]}" -DET_F32_TENSOR_TESTING=1 \
    -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 "${sources[@]}" -lm \
    -o "$out/$mode/test" >"$out/$mode/compile.stdout" \
    2>"$out/$mode/compile.stderr"
  [[ ! -s "$out/$mode/compile.stderr" ]]
  set +e
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 90s \
    "$out/$mode/test" >"$out/$mode/run.stdout" 2>"$out/$mode/run.stderr"
  status=$?
  set -e
  printf '%s\n' "$status" >"$out/$mode/run.exit"
  [[ "$status" -eq 0 && ! -s "$out/$mode/run.stderr" ]]
  grep -E '^G3-R M3T parameter preflight PASS: [0-9]+ checks$' \
    "$out/$mode/run.stdout" >/dev/null
done
cmp "$out/normal/run.stdout" "$out/repeat/run.stdout"
cmp "$out/normal/run.stdout" "$out/san/run.stdout"
printf 'G3-R M3T parameter preflight gate PASS\n'
