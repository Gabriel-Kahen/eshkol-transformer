#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d /tmp/g3r-held-image-host.XXXXXX)"
trap 'rm -rf -- "$out"' EXIT
cc="${CC:-cc}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -I "$root/src"
       -DET_G3R_COMMITTED_HISTORY_PRIVATE -DET_G3R_SAME_OWNER_SAVE_PRIVATE)
"$cc" "${flags[@]}" "$root/native/g3r_held_image_bridge.c" \
  "$root/tests/g3r/held_image_bridge_test.c" -o "$out/bridge"
"$out/bridge"
"$cc" "${flags[@]}" -c "$root/native/g3r_held_image_bridge.c" \
  -o "$out/on.o"
nm -g --defined-only "$out/on.o" | rg -q ' et_g3r_private_held_image_copy_v1$'
if "$cc" -std=c11 -I "$root/src" \
    -c "$root/native/g3r_held_image_bridge.c" -o "$out/off.o" \
    >"$out/off.stdout" 2>"$out/off.stderr"; then
  echo 'feature-off bridge unexpectedly compiled' >&2
  exit 1
fi
PYTHONDONTWRITEBYTECODE=1 python3 "$root/tests/g3r/test_held_image_adapter_source.py"
"$root/scripts/test-g3r-o2-held-save-native.sh"
"$root/scripts/test-g3r-current-history-held-owner.sh"
echo 'G3R HELD IMAGE HOST PASS'
