#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-hold-bridge.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
cc="${CC:-clang-22}"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow
  -I "$root/include" -I "$root/native" -I "$root/src"
  -I "$root/src/eshkol_transformer")
feature=(-DET_G3R_SAME_OWNER_SAVE_PRIVATE
  -DET_G3R_COMMITTED_HISTORY_PRIVATE)
"$cc" "${common[@]}" "${feature[@]}" \
  "$root/native/g3r_same_owner_hold_bridge.c" \
  "$root/tests/g3r/same_owner_hold_bridge_test.c" -o "$out/normal"
"$out/normal" >"$out/normal.out"
"$out/normal" >"$out/repeat.out"
cmp "$out/normal.out" "$out/repeat.out"
"$cc" "${common[@]}" "${feature[@]}" -g \
  -fsanitize=address,undefined \
  "$root/native/g3r_same_owner_hold_bridge.c" \
  "$root/tests/g3r/same_owner_hold_bridge_test.c" -o "$out/san"
ASAN_OPTIONS=detect_leaks=1 "$out/san" >"$out/san.out"
cmp "$out/normal.out" "$out/san.out"
"$cc" "${common[@]}" -c \
  "$root/native/g3r_same_owner_hold_bridge.c" -o "$out/off.o"
if nm -g --defined-only "$out/off.o" | \
    grep -F 'et_g3r_private_same_owner_holds_'; then
  echo 'same-owner hold bridge escaped feature-off object' >&2
  exit 1
fi
grep -Fx 'same-owner hold bridge fixture PASS' "$out/normal.out" >/dev/null
echo 'same-owner hold bridge PASS (normal, repeat, sanitizer, feature off)'
