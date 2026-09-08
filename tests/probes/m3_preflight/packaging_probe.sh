#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
out="$(mktemp -d "${TMPDIR:-/tmp}/m3-packaging.XXXXXX")"
trap 'rm -rf -- "${out}"' EXIT
cc="${PACKAGING_CC:-/usr/bin/clang}"
printf 'Diagnostic host-native build only; no supported-lane or AOT claim.\n'
"${cc}" --version
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -O2
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -frounding-math -I include -I native)
objects=()
for source in kernel_abi i64_tensor f32_tensor n2_primitives_provider \
    a2_attention_provider a2_kv_cache indexed_cross_entropy; do
  "${cc}" "${flags[@]}" -c "native/${source}.c" -o "${out}/${source}.o"
  objects+=("${out}/${source}.o")
done
if nm -g --defined-only "${objects[@]}" |
    awk '$NF == "eshkol_transformer_kernel_provider_v1" { found=1 } END {exit !found}'; then
  printf 'FAIL: canonical resolver definition found\n' >&2
  exit 1
fi
"${cc}" "${flags[@]}" tests/probes/m3_preflight/packaging_provider_inventory.c \
  "${objects[@]}" -lm -o "${out}/inventory"
"${out}/inventory" >"${out}/first.stdout"
"${out}/inventory" >"${out}/second.stdout"
cmp "${out}/first.stdout" "${out}/second.stdout"
cat "${out}/first.stdout"
"${cc}" "${flags[@]}" tests/n2/test_i2_integration.c "${objects[@]}" \
  -lm -o "${out}/n2-i2"
"${out}/n2-i2"
"${cc}" "${flags[@]}" tests/k1/test_kernel_abi.c "${objects[@]}" \
  -lm -o "${out}/k1"
"${out}/k1"
