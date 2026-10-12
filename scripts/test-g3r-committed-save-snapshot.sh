#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-save-snapshot.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
python3 "$root/scripts/check-g3r-committed-save-snapshot.py"
cc -std=c11 -O0 -Wall -Wextra -Werror -Wno-misleading-indentation \
  -I "$root/include" -I "$root/native" -I "$root/src" \
  -I "$root/src/eshkol_transformer" \
  -c "$root/src/eshkol_transformer/g3t_transport.c" -o "$out/off.o"
if nm -g --defined-only "$out/off.o" | grep -Fq \
    et_g3t_private_committed_save_copy_v1; then
  echo "G3-R save snapshot escaped feature-off object" >&2
  exit 1
fi
for mode in normal repeat sanitizer; do
  flags=(-O0)
  if [[ "$mode" == sanitizer ]]; then
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  cc -std=c11 -Wall -Wextra -Werror -Wno-unused-parameter \
    -Wno-misleading-indentation -ffunction-sections -fdata-sections \
    "${flags[@]}" -I "$root/include" -I "$root/native" -I "$root/src" \
    -I "$root/src/eshkol_transformer" \
    "$root/tests/g3r/committed_save_snapshot_test.c" \
    -Wl,--gc-sections -o "$out/$mode"
  if [[ "$mode" == sanitizer ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 "$out/$mode"
  else
    "$out/$mode"
  fi
done
