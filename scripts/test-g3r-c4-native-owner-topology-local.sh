#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
cc="${CC:-clang}"
tmp="$(mktemp -d)"
trap 'rm -rf -- "${tmp}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -g -O1
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I include -I native -I src/eshkol_transformer)
sources=(tests/g3c4/test_sealed_owner_topology.c
  tests/m3t/kernel_fail_allocator.c native/n3k_primitives_provider.c)

"${cc}" "${flags[@]}" -DET_F32_TENSOR_TESTING -DET_M3_CALL_TESTING \
  "${sources[@]}" -lm -o "${tmp}/normal"
"${tmp}/normal" >"${tmp}/normal.stdout"
"${tmp}/normal" >"${tmp}/repeat.stdout"
cmp "${tmp}/normal.stdout" "${tmp}/repeat.stdout"
"${cc}" "${flags[@]}" -fsanitize=address,undefined \
  -fno-omit-frame-pointer -DET_F32_TENSOR_TESTING -DET_M3_CALL_TESTING \
  "${sources[@]}" -lm -o "${tmp}/san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 "${tmp}/san" >"${tmp}/san.stdout"
cmp "${tmp}/normal.stdout" "${tmp}/san.stdout"

"${cc}" "${flags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE -c \
  src/eshkol_transformer/g3c4_model_owner.c -o "${tmp}/off.o"
"${cc}" "${flags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -DET_G3C4_NATIVE_PINS_PRIVATE -DET_G3R_C4_OWNER_TOPOLOGY_PRIVATE \
  -DET_G3R_C4_OWNER_ASSOCIATION_PRIVATE -c \
  src/eshkol_transformer/g3c4_model_owner.c -o "${tmp}/on.o"
for symbol in et_g3c4_private_sealed_owner_topology_status_v1 \
              et_g3c4_private_sealed_owner_binding_status_v1; do
  if nm -g --defined-only "${tmp}/off.o" | awk '{print $3}' | rg -Fx "${symbol}"; then
    echo "feature-off owner exports ${symbol}" >&2
    exit 1
  fi
  nm -g --defined-only "${tmp}/on.o" | awk '{print $3}' | rg -Fx "${symbol}" >/dev/null
done
cat "${tmp}/normal.stdout"
