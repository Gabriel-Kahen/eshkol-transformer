#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-o2-ticket-adopter.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
cc="${CC:-clang-22}"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffunction-sections -fdata-sections
  -I "$root/include" -I "$root/native")
feature=(-DET_O2_NATIVE_HELPERS_ONLY -DET_G3R_SAME_OWNER_SAVE_PRIVATE)
for mode in normal repeat sanitizer; do
  flags=(-O0)
  if [[ "$mode" == sanitizer ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${common[@]}" "${feature[@]}" "${flags[@]}" \
    "$root/native/o2_wave2_package_bridge.c" \
    "$root/tests/g3r/held_o2_ticket_adopter_test.c" \
    -Wl,--gc-sections -o "$out/$mode"
  ASAN_OPTIONS=detect_leaks=1 "$out/$mode" >"$out/$mode.out"
  grep -Fx 'held O2 ticket adopter PASS' "$out/$mode.out" >/dev/null
done
cmp "$out/normal.out" "$out/repeat.out"
cmp "$out/normal.out" "$out/sanitizer.out"
"$cc" "${common[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -c "$root/native/o2_wave2_package_bridge.c" -o "$out/off.o"
if nm -g --defined-only "$out/off.o" | \
    grep -F 'et_g3r_private_o2_adopter_'; then
  echo 'ticket adopter escaped feature-off object' >&2
  exit 1
fi
echo 'held O2 ticket adopter PASS (normal, repeat, sanitizer, feature off)'
