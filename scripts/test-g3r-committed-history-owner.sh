#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-history-owner.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
python3 "$root/scripts/check-g3r-committed-history-owner.py"
cc -std=c11 -O0 -Wall -Wextra -Werror -Wno-misleading-indentation \
  -I "$root/include" -I "$root/native" -I "$root/src" \
  -I "$root/src/eshkol_transformer" \
  -c "$root/src/eshkol_transformer/g3t_transport.c" \
  -o "$out/feature-off.o"
nm -g --defined-only "$out/feature-off.o" | awk '{print $3}' \
  > "$out/feature-off-symbols"
if grep -Fxq et_g3t_private_committed_history_snapshot_v1 \
    "$out/feature-off-symbols"; then
  echo "G3-R history snapshot escaped feature-off object" >&2
  exit 1
fi
echo "G3-R committed history feature-off symbol closure PASS"
cc -std=c11 -O0 -Wall -Wextra -Werror -Wno-misleading-indentation \
  -ffunction-sections -fdata-sections -I "$root/include" -I "$root/native" \
  -I "$root/src" -I "$root/src/eshkol_transformer" \
  "$root/tests/g3r/committed_history_owner_test.c" -Wl,--gc-sections \
  -o "$out/test"
"$out/test"
cc -std=c11 -O0 -Wall -Wextra -Werror -Wno-unused-parameter \
  -Wno-misleading-indentation -ffunction-sections -fdata-sections \
  -I "$root/include" -I "$root/native" -I "$root/src" \
  -I "$root/src/eshkol_transformer" \
  "$root/tests/g3r/committed_history_transport_test.c" \
  -Wl,--gc-sections -o "$out/transport"
"$out/transport"
