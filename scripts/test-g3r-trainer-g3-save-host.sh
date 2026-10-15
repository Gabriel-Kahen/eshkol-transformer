#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
python3 "$root/tests/g3r/test_trainer_g3_save_source.py"
if cc -std=c11 -I "$root/include" -I "$root/src" -I "$root/native" \
  -c "$root/native/g3r_trainer_g3_save_bridge.c" \
  -o "$tmp/feature-off.o" 2>"$tmp/feature-off.stderr"; then
  echo 'private trainer G3 bridge compiled without feature contracts' >&2
  exit 1
fi
grep -F 'requires the held G3/M3T feature contracts' \
  "$tmp/feature-off.stderr" >/dev/null
cc -std=c11 -Wall -Wextra -Werror \
  -DET_G3R_SAME_OWNER_SAVE_PRIVATE -DET_G3R_COMMITTED_HISTORY_PRIVATE \
  -I "$root/include" -I "$root/src" -I "$root/native" \
  "$root/native/g3r_trainer_g3_save_bridge.c" \
  "$root/tests/g3r/trainer_g3_save_bridge_test.c" -o "$tmp/bridge-test"
"$tmp/bridge-test"
