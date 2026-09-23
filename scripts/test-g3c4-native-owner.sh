#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in comm nm; do require_command "${command}"; done
owner_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
owner_evidence="${1:-$(project_build_dir)/g3c4-native-owner-test}"
mkdir -p "${owner_evidence}"

owner_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src/eshkol_transformer"
)

"${owner_cc}" "${owner_flags[@]}" -O2 \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" \
  -o "${owner_evidence}/ordinary-f32.o"
"${owner_cc}" "${owner_flags[@]}" -O2 -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" \
  -o "${owner_evidence}/c4-private-f32.o"
nm -g --defined-only --format=posix "${owner_evidence}/ordinary-f32.o" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"${owner_evidence}/ordinary-defined.txt"
nm -g --defined-only --format=posix "${owner_evidence}/c4-private-f32.o" |
  awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
  >"${owner_evidence}/c4-private-defined.txt"
comm -23 "${owner_evidence}/ordinary-defined.txt" \
  "${owner_evidence}/c4-private-defined.txt" \
  >"${owner_evidence}/unexpected-ordinary-only.txt"
comm -13 "${owner_evidence}/ordinary-defined.txt" \
  "${owner_evidence}/c4-private-defined.txt" \
  >"${owner_evidence}/c4-private-added.txt"
test ! -s "${owner_evidence}/unexpected-ordinary-only.txt"
printf '%s\n' et_f32_parameter_idle_preflight_internal |
  cmp - "${owner_evidence}/c4-private-added.txt"

owner_sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_model_owner.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
)
for owner_mode in normal sanitize; do
  owner_mode_flags=(-O2)
  owner_environment=()
  if [[ "${owner_mode}" == sanitize ]]; then
    owner_mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    owner_environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "${owner_cc}" "${owner_flags[@]}" "${owner_mode_flags[@]}" \
    "${owner_sources[@]}" -lm -o "${owner_evidence}/owner-${owner_mode}"
  "${owner_environment[@]}" "${owner_evidence}/owner-${owner_mode}" \
    >"${owner_evidence}/owner-${owner_mode}.stdout" \
    2>"${owner_evidence}/owner-${owner_mode}.stderr"
  test ! -s "${owner_evidence}/owner-${owner_mode}.stderr"
  grep -E '^G3-C4 seeded native owner: [0-9]+ checks$' \
    "${owner_evidence}/owner-${owner_mode}.stdout" >/dev/null
done
cmp "${owner_evidence}/owner-normal.stdout" \
  "${owner_evidence}/owner-sanitize.stdout"
cat "${owner_evidence}/owner-normal.stdout"
printf 'G3-C4 native-owner evidence: %s\n' "${owner_evidence}"
