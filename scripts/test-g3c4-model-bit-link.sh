#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evidence="${1:-${root}/build/g3c4-model-bit-link}"
cc="${CC:-clang}"
mkdir -p "${evidence}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -O2
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fvisibility=hidden -I "${root}/include" -I "${root}/native"
  -I "${root}/src/eshkol_transformer")
f32="${root}/src/eshkol_transformer/m3t_f32_integration.c"
owner="${root}/src/eshkol_transformer/g3c4_model_owner.c"

"${cc}" "${flags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${f32}" -o "${evidence}/f32-off.o"
"${cc}" "${flags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -DET_G3C4_MODEL_BIT_PRIVATE -c "${f32}" \
  -o "${evidence}/f32-on.o"
"${cc}" "${flags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${owner}" -o "${evidence}/owner-off.o"
"${cc}" "${flags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -DET_G3C4_MODEL_BIT_PRIVATE -c "${owner}" \
  -o "${evidence}/owner-on.o"

for unit in f32-off f32-on owner-off owner-on; do
  nm -g --defined-only --format=posix "${evidence}/${unit}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${unit}.defined"
  nm -u --format=posix "${evidence}/${unit}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${unit}.undefined"
done
comm -23 "${evidence}/f32-off.defined" \
  "${evidence}/f32-on.defined" >"${evidence}/f32-lost.defined"
comm -23 "${evidence}/owner-off.defined" \
  "${evidence}/owner-on.defined" >"${evidence}/owner-lost.defined"
test ! -s "${evidence}/f32-lost.defined"
test ! -s "${evidence}/owner-lost.defined"
printf '%s\n' et_f32_private_output_span_disjoint_internal |
  cmp - <(comm -13 "${evidence}/f32-off.defined" \
    "${evidence}/f32-on.defined")
printf '%s\n' et_g3c4_private_model_owner_binding_matches_v1 \
  et_g3c4_private_model_owner_copy_bits_v1 |
  cmp - <(comm -13 "${evidence}/owner-off.defined" \
    "${evidence}/owner-on.defined")
grep -Fx et_f32_private_output_span_disjoint_internal \
  "${evidence}/owner-on.undefined" >/dev/null
"${cc}" "${flags[@]}" -DET_G3C4_MODEL_BIT_PRIVATE \
  "${root}/tests/g3c4/test_model_bit_link.c" \
  "${evidence}/owner-on.o" "${evidence}/f32-on.o" \
  "${root}/native/n3k_primitives_provider.c" \
  "${root}/tests/m3t/kernel_fail_allocator.c" -lm \
  -o "${evidence}/model-bit-link"
"${evidence}/model-bit-link" >"${evidence}/model-bit-link.stdout" \
  2>"${evidence}/model-bit-link.stderr"
test ! -s "${evidence}/model-bit-link.stderr"
printf '%s\n' 'C4 model-bit separate-TU link: passed' |
  cmp - "${evidence}/model-bit-link.stdout"
cat "${evidence}/model-bit-link.stdout"
