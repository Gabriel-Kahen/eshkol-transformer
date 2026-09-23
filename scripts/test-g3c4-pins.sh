#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in comm git nm tar; do require_command "${command}"; done
pins_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
pins_evidence="${1:-$(project_build_dir)/g3c4-pins-test}"
mkdir -p "${pins_evidence}/base"

pins_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src/eshkol_transformer"
)

git -C "${PROJECT_ROOT}" archive \
  d5fa9a71b000ab1139a25e9b734b230ab5bda759 | \
  tar -x -C "${pins_evidence}/base"
"${pins_cc}" "${pins_flags[@]}" -O2 \
  -c "${pins_evidence}/base/src/eshkol_transformer/m3_call_f32_integration.c" \
  -o "${pins_evidence}/base-ordinary.o"
"${pins_cc}" "${pins_flags[@]}" -O2 \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/m3_call_f32_integration.c" \
  -o "${pins_evidence}/ordinary.o"
cmp "${pins_evidence}/base-ordinary.o" "${pins_evidence}/ordinary.o"

"${pins_cc}" "${pins_flags[@]}" -O2 -DET_G3C4_NATIVE_PINS_PRIVATE \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/m3_call_f32_integration.c" \
  -o "${pins_evidence}/c4-private.o"
nm -g --defined-only --format=posix "${pins_evidence}/ordinary.o" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"${pins_evidence}/ordinary-defined.txt"
nm -g --defined-only --format=posix "${pins_evidence}/c4-private.o" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"${pins_evidence}/c4-private-defined.txt"
comm -23 "${pins_evidence}/ordinary-defined.txt" \
  "${pins_evidence}/c4-private-defined.txt" \
  >"${pins_evidence}/unexpected-ordinary-only.txt"
comm -13 "${pins_evidence}/ordinary-defined.txt" \
  "${pins_evidence}/c4-private-defined.txt" \
  >"${pins_evidence}/c4-private-added.txt"
test ! -s "${pins_evidence}/unexpected-ordinary-only.txt"
printf '%s\n' et_g3c4_model_pins_begin_internal \
  et_g3c4_model_pins_check_internal et_g3c4_model_pins_end_internal |
  LC_ALL=C sort | cmp - "${pins_evidence}/c4-private-added.txt"

pins_sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_pins.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
)
c2_sources=(
  "${PROJECT_ROOT}/tests/m3cg/test_pins.c"
  "${PROJECT_ROOT}/native/kernel_abi.c"
)
for pins_mode in normal sanitize; do
  pins_mode_flags=(-O2)
  pins_environment=()
  if [[ "${pins_mode}" == sanitize ]]; then
    pins_mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    pins_environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "${pins_cc}" "${pins_flags[@]}" "${pins_mode_flags[@]}" \
    -DET_F32_TENSOR_TESTING -DET_M3_CALL_TESTING \
    "${pins_sources[@]}" -lm -o "${pins_evidence}/pins-${pins_mode}"
  "${pins_environment[@]}" "${pins_evidence}/pins-${pins_mode}" \
    >"${pins_evidence}/pins-${pins_mode}.stdout" \
    2>"${pins_evidence}/pins-${pins_mode}.stderr"
  test ! -s "${pins_evidence}/pins-${pins_mode}.stderr"
  grep -F 'g3c4 pins: genuine owner, geometry split, 98 busy controls,' \
    "${pins_evidence}/pins-${pins_mode}.stdout" >/dev/null

  "${pins_cc}" "${pins_flags[@]}" "${pins_mode_flags[@]}" \
    -DET_F32_TENSOR_TESTING -DET_M3_CALL_TESTING \
    "${c2_sources[@]}" -lm -o "${pins_evidence}/c2-${pins_mode}"
  "${pins_environment[@]}" "${pins_evidence}/c2-${pins_mode}" \
    >"${pins_evidence}/c2-${pins_mode}.stdout" \
    2>"${pins_evidence}/c2-${pins_mode}.stderr"
  test ! -s "${pins_evidence}/c2-${pins_mode}.stderr"

  "${pins_cc}" "${pins_flags[@]}" "${pins_mode_flags[@]}" \
    -DET_F32_TENSOR_TESTING -DET_M3_CALL_TESTING \
    -DET_G3C4_NATIVE_PINS_PRIVATE \
    "${c2_sources[@]}" -lm -o "${pins_evidence}/c2-private-${pins_mode}"
  "${pins_environment[@]}" "${pins_evidence}/c2-private-${pins_mode}" \
    >"${pins_evidence}/c2-private-${pins_mode}.stdout" \
    2>"${pins_evidence}/c2-private-${pins_mode}.stderr"
  test ! -s "${pins_evidence}/c2-private-${pins_mode}.stderr"
done
"${pins_evidence}/pins-normal" >"${pins_evidence}/pins-repeat.stdout" \
  2>"${pins_evidence}/pins-repeat.stderr"
test ! -s "${pins_evidence}/pins-repeat.stderr"
cmp "${pins_evidence}/pins-normal.stdout" \
  "${pins_evidence}/pins-repeat.stdout"
cmp "${pins_evidence}/pins-normal.stdout" \
  "${pins_evidence}/pins-sanitize.stdout"
cmp "${pins_evidence}/c2-normal.stdout" \
  "${pins_evidence}/c2-sanitize.stdout"
cmp "${pins_evidence}/c2-normal.stdout" \
  "${pins_evidence}/c2-private-normal.stdout"
cmp "${pins_evidence}/c2-private-normal.stdout" \
  "${pins_evidence}/c2-private-sanitize.stdout"
cat "${pins_evidence}/pins-normal.stdout"
printf 'G3-C4 pin evidence: %s\n' "${pins_evidence}"
