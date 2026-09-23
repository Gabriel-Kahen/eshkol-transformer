#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in comm git nm tar; do require_command "${command}"; done
bridge_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
bridge_evidence="${1:-$(project_build_dir)/g3c4-i2-construction-bridge-test}"
mkdir -p "${bridge_evidence}/base"

bridge_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src/eshkol_transformer"
)
base_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${bridge_evidence}/base/include"
  -I "${bridge_evidence}/base/native"
  -I "${bridge_evidence}/base/src/eshkol_transformer"
)

git -C "${PROJECT_ROOT}" archive \
  4d4fafcc083f464f5a86dca388ef4b93d1f32e9a | \
  tar -x -C "${bridge_evidence}/base"
"${bridge_cc}" "${base_flags[@]}" -O2 -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${bridge_evidence}/base/native/i2_wave2_package_bridge.c" \
  -o "${bridge_evidence}/base-ordinary.o"
"${bridge_cc}" "${bridge_flags[@]}" -O2 -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${bridge_evidence}/ordinary.o"
cmp "${bridge_evidence}/base-ordinary.o" "${bridge_evidence}/ordinary.o"

"${bridge_cc}" "${bridge_flags[@]}" -O2 -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_G3C4_I2_CONSTRUCTION_PRIVATE \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${bridge_evidence}/surface.o"
"${bridge_cc}" "${bridge_flags[@]}" -O2 -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_G3C4_I2_CONSTRUCTION_PRIVATE -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${bridge_evidence}/source.o"

for object in ordinary surface source; do
  nm -g --defined-only --format=posix "${bridge_evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${bridge_evidence}/${object}-defined.txt"
  nm -g --undefined-only --format=posix "${bridge_evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${bridge_evidence}/${object}-undefined.txt"
done
comm -23 "${bridge_evidence}/ordinary-defined.txt" \
  "${bridge_evidence}/surface-defined.txt" \
  >"${bridge_evidence}/unexpected-ordinary-only.txt"
comm -13 "${bridge_evidence}/ordinary-defined.txt" \
  "${bridge_evidence}/surface-defined.txt" \
  >"${bridge_evidence}/surface-added.txt"
test ! -s "${bridge_evidence}/unexpected-ordinary-only.txt"
printf '%s\n' et_i2_private_g3c4_construction_available_v1 \
  et_i2_private_g3c4_construction_parameter_preflight_v1 |
  LC_ALL=C sort | cmp - "${bridge_evidence}/surface-added.txt"
cmp "${bridge_evidence}/surface-defined.txt" \
  "${bridge_evidence}/source-defined.txt"

comm -13 "${bridge_evidence}/ordinary-undefined.txt" \
  "${bridge_evidence}/surface-undefined.txt" \
  >"${bridge_evidence}/surface-added-undefined.txt"
test ! -s "${bridge_evidence}/surface-added-undefined.txt"
comm -13 "${bridge_evidence}/surface-undefined.txt" \
  "${bridge_evidence}/source-undefined.txt" \
  >"${bridge_evidence}/source-added-undefined.txt"
printf '%s\n' et_g3c4_construction_parameter_preflight_internal |
  cmp - "${bridge_evidence}/source-added-undefined.txt"

genuine_sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_i2_construction_bridge.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
)
unavailable_sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_i2_construction_unavailable.c"
  "${PROJECT_ROOT}/native/kernel_abi.c"
)
for bridge_mode in normal sanitize; do
  bridge_mode_flags=(-O2)
  bridge_environment=()
  if [[ "${bridge_mode}" == sanitize ]]; then
    bridge_mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    bridge_environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "${bridge_cc}" "${bridge_flags[@]}" "${bridge_mode_flags[@]}" \
    "${genuine_sources[@]}" -lm -o "${bridge_evidence}/genuine-${bridge_mode}"
  "${bridge_environment[@]}" "${bridge_evidence}/genuine-${bridge_mode}" \
    >"${bridge_evidence}/genuine-${bridge_mode}.stdout" \
    2>"${bridge_evidence}/genuine-${bridge_mode}.stderr"
  test ! -s "${bridge_evidence}/genuine-${bridge_mode}.stderr"
  grep -F 'G3-C4 I2 construction bridge: availability, lifecycle, 98 busy controls, exact diagnostics passed' \
    "${bridge_evidence}/genuine-${bridge_mode}.stdout" >/dev/null

  "${bridge_cc}" "${bridge_flags[@]}" "${bridge_mode_flags[@]}" \
    "${unavailable_sources[@]}" -lm \
    -o "${bridge_evidence}/unavailable-${bridge_mode}"
  "${bridge_environment[@]}" "${bridge_evidence}/unavailable-${bridge_mode}" \
    >"${bridge_evidence}/unavailable-${bridge_mode}.stdout" \
    2>"${bridge_evidence}/unavailable-${bridge_mode}.stderr"
  test ! -s "${bridge_evidence}/unavailable-${bridge_mode}.stderr"
done
"${bridge_evidence}/genuine-normal" \
  >"${bridge_evidence}/genuine-repeat.stdout" \
  2>"${bridge_evidence}/genuine-repeat.stderr"
"${bridge_evidence}/unavailable-normal" \
  >"${bridge_evidence}/unavailable-repeat.stdout" \
  2>"${bridge_evidence}/unavailable-repeat.stderr"
test ! -s "${bridge_evidence}/genuine-repeat.stderr"
test ! -s "${bridge_evidence}/unavailable-repeat.stderr"
cmp "${bridge_evidence}/genuine-normal.stdout" \
  "${bridge_evidence}/genuine-repeat.stdout"
cmp "${bridge_evidence}/genuine-normal.stdout" \
  "${bridge_evidence}/genuine-sanitize.stdout"
cmp "${bridge_evidence}/unavailable-normal.stdout" \
  "${bridge_evidence}/unavailable-repeat.stdout"
cmp "${bridge_evidence}/unavailable-normal.stdout" \
  "${bridge_evidence}/unavailable-sanitize.stdout"

"${PROJECT_ROOT}/scripts/test-g3c4-native-owner.sh" \
  "${bridge_evidence}/owner" >"${bridge_evidence}/owner-run.stdout"
"${PROJECT_ROOT}/scripts/test-g3c4-pins.sh" \
  "${bridge_evidence}/pins" >"${bridge_evidence}/pins-run.stdout"
"${PROJECT_ROOT}/scripts/test-m3t-construction.sh" \
  >"${bridge_evidence}/m3t-construction.stdout"

cat "${bridge_evidence}/genuine-normal.stdout"
cat "${bridge_evidence}/unavailable-normal.stdout"
cat "${bridge_evidence}/m3t-construction.stdout"
printf 'G3-C4 I2 construction bridge evidence: %s\n' "${bridge_evidence}"
