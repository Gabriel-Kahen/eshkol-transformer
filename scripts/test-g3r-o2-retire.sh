#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
baseline=af987fe2318ad205f6de5e78cd141b3c4265f8b4
if [[ $# -ne 2 || "$1" != /* || "$2" != /* || ! -x "$1" || -e "$2" || -L "$2" ]]; then
  echo 'usage: test-g3r-o2-retire.sh /absolute/compiler /fresh/absolute/output' >&2
  exit 2
fi
cc="$1"
out="$2"
mkdir -p -- "$out/base" "$out/off" "$out/normal" "$out/repeat" "$out/san"
git -C "$root" show "$baseline:native/o2_optimizer.c" >"$out/base/o2_optimizer.c"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -I "$root/include" -I "$root/native")
"$cc" "${flags[@]}" -c "$out/base/o2_optimizer.c" -o "$out/base/o2.o" \
  >"$out/base/compile.stdout" 2>"$out/base/compile.stderr"
"$cc" "${flags[@]}" -c "$root/native/o2_optimizer.c" -o "$out/off/o2.o" \
  >"$out/off/compile.stdout" 2>"$out/off/compile.stderr"
cmp "$out/base/o2.o" "$out/off/o2.o"
[[ ! -s "$out/base/compile.stderr" && ! -s "$out/off/compile.stderr" ]]
nm -g --defined-only --format=posix "$out/off/o2.o" >"$out/off/symbols"
! grep -F 'et_o2_private_candidate_retire_' "$out/off/symbols"

"$cc" "${flags[@]}" -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -c "$root/native/o2_optimizer.c" \
  -o "$out/off/on.o" >"$out/off/on.compile.stdout" \
  2>"$out/off/on.compile.stderr"
[[ ! -s "$out/off/on.compile.stderr" ]]
nm -u "$out/off/on.o" >"$out/off/on.undefined"
grep -F 'et_f32_tensor_private_owned_span_overlap_v1' \
  "$out/off/on.undefined" >/dev/null
! grep -F 'et_f32_tensor_private_storage_overlap_v1' \
  "$out/off/on.undefined"
nm -g --defined-only --format=posix "$out/off/on.o" | cut -d' ' -f1 | \
  sort >"$out/off/on.names"
cut -d' ' -f1 "$out/off/symbols" | sort >"$out/off/off.names"
comm -13 "$out/off/off.names" "$out/off/on.names" >"$out/off/private.delta"
comm -23 "$out/off/off.names" "$out/off/on.names" >"$out/off/removed.names"
cat >"$out/off/expected.delta" <<'EOF'
et_o2_private_candidate_retire_commit_v1
et_o2_private_candidate_retire_preflight_v1
EOF
cmp "$out/off/expected.delta" "$out/off/private.delta"
[[ ! -s "$out/off/removed.names" ]]

cat >"$out/off/header-on.c" <<'EOF'
#include "o2_optimizer_internal.h"
int32_t (*private_preflight)(et_o2_optimizer *, et_o2_error_v1 *) =
    &et_o2_private_candidate_retire_preflight_v1;
int32_t (*private_commit)(et_o2_optimizer *, et_o2_error_v1 *) =
    &et_o2_private_candidate_retire_commit_v1;
EOF
"$cc" "${flags[@]}" -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1 \
  -fsyntax-only "$out/off/header-on.c" >"$out/off/header-on.stdout" \
  2>"$out/off/header-on.stderr"
[[ ! -s "$out/off/header-on.stderr" ]]
cat >"$out/off/header-off.c" <<'EOF'
#include "o2_optimizer_internal.h"
int32_t (*forbidden)(et_o2_optimizer *, et_o2_error_v1 *) =
    &et_o2_private_candidate_retire_preflight_v1;
EOF
set +e
"$cc" "${flags[@]}" -fsyntax-only "$out/off/header-off.c" \
  >"$out/off/header-off.stdout" 2>"$out/off/header-off.stderr"
header_status=$?
set -e
printf '%d\n' "$header_status" >"$out/off/header-off.exit"
[[ "$header_status" -eq 1 ]]
grep -F "undeclared identifier 'et_o2_private_candidate_retire_preflight_v1'" \
  "$out/off/header-off.stderr" >/dev/null

for mode in normal repeat san; do
  mode_flags=("${flags[@]}" -DET_G3R_CANDIDATE_RETIRE_PRIVATE=1
    -DET_F32_TENSOR_TESTING=1 -DET_TR3_O2_STEP_CLEAR_NATIVE=1)
  if [[ "$mode" == san ]]; then
    mode_flags+=(-g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined,leak)
  fi
  "$cc" "${mode_flags[@]}" "$root/native/f32_tensor.c" \
    "$root/native/kernel_abi.c" "$root/tests/o2/test_g3r_o2_retire.c" \
    -o "$out/$mode/test" >"$out/$mode/compile.stdout" \
    2>"$out/$mode/compile.stderr"
  [[ ! -s "$out/$mode/compile.stderr" ]]
  nm -g --defined-only --format=posix "$out/$mode/test" \
    >"$out/$mode/symbols"
  grep -F 'et_o2_private_candidate_retire_preflight_v1 ' \
    "$out/$mode/symbols" >/dev/null
  grep -F 'et_o2_private_candidate_retire_commit_v1 ' \
    "$out/$mode/symbols" >/dev/null
  set +e
  ASAN_OPTIONS=detect_leaks=1 timeout --foreground --signal=TERM --kill-after=5s \
    90s "$out/$mode/test" >"$out/$mode/run.stdout" 2>"$out/$mode/run.stderr"
  status=$?
  set -e
  printf '%s\n' "$status" >"$out/$mode/run.exit"
  [[ "$status" -eq 0 && ! -s "$out/$mode/run.stderr" ]]
  grep -Fx 'G3R O2 retirement PASS: 1812 checks' \
    "$out/$mode/run.stdout" >/dev/null
done
cmp "$out/normal/run.stdout" "$out/repeat/run.stdout"
cmp "$out/normal/run.stdout" "$out/san/run.stdout"
printf 'G3R O2 retirement gate PASS\n'
