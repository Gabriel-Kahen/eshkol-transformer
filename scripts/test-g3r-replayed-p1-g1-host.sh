#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/g3r-replay-host.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
PYTHONDONTWRITEBYTECODE=1 python3 "$root/scripts/check-g3r-replayed-p1-g1.py"
common=(-std=c11 -O0 -Wall -Wextra -Werror -Wno-misleading-indentation
  -I "$root/include" -I "$root/native" -I "$root/src"
  -I "$root/src/eshkol_transformer")
cc "${common[@]}" -c "$root/src/eshkol_transformer/g3t_transport.c" -o "$out/off.o"
if nm -g --defined-only "$out/off.o" | grep -Eq 'et_g3t_private_replayed_p1_g1_(preflight|acquire)_v1'; then
  echo 'private replay ABI leaked in feature-off object' >&2
  exit 1
fi
cc "${common[@]}" -ffunction-sections -fdata-sections \
  "$root/tests/g3r/replayed_p1_g1_native_test.c" -Wl,--gc-sections \
  -o "$out/native"
"$out/native"
echo 'G3-R replayed P1/G1 host checks PASS; compiled Eshkol gate unrun'
