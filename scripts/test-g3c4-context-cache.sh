#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp comm git nm tar; do require_command "${command}"; done
context_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
evidence="${1:-$(project_build_dir)/g3c4-context-cache-test}"
mkdir -p "${evidence}/base"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)

git -C "${PROJECT_ROOT}" archive 9d4f97af37c54cc84bf6205c46c3090c32959ba3 |
  tar -x -C "${evidence}/base"
"${context_cc}" "${flags[@]}" -O2 -c \
  "${evidence}/base/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/base-owner.o"
"${context_cc}" "${flags[@]}" -O2 -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/ordinary-owner.o"
cmp "${evidence}/base-owner.o" "${evidence}/ordinary-owner.o"

"${context_cc}" "${flags[@]}" -O2 -DET_G3C4_CONTEXT_PRIVATE -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/context-owner.o"
for object in ordinary-owner context-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/ordinary-owner-defined.txt" \
  "${evidence}/context-owner-defined.txt" >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_context_close_v1 \
  et_g3c4_private_context_create_v1 | LC_ALL=C sort |
  cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/ordinary-owner-undefined.txt" \
  "${evidence}/context-owner-undefined.txt" >"${evidence}/added-undefined.txt"
printf '%s\n' et_a2_kv_cache_create_v1 et_a2_kv_cache_destroy_v1 |
  LC_ALL=C sort | cmp - "${evidence}/added-undefined.txt"

cat >"${evidence}/header-c.c" <<'EOF'
#include "eshkol_transformer/g3c4_context_internal.h"
int main(void) { return 0; }
EOF
cat >"${evidence}/header-cpp.cpp" <<'EOF'
#include "eshkol_transformer/g3c4_context_internal.h"
int main() { return 0; }
EOF
"${context_cc}" "${flags[@]}" -c "${evidence}/header-c.c" \
  -o "${evidence}/header-c.o"
context_cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
"${context_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/src" -c "${evidence}/header-cpp.cpp" \
  -o "${evidence}/header-cpp.o"

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_context_cache.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
)
for mode in normal sanitize; do
  mode_flags=(-O2)
  environment=()
  if [[ "${mode}" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "${context_cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_G3C4_A2_INTRUSIVE_TU \
    -c "${PROJECT_ROOT}/tests/g3c4/test_context_cache.c" \
    -o "${evidence}/context-a2-${mode}.o"
  "${context_cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING "${sources[@]}" \
    "${evidence}/context-a2-${mode}.o" -lm \
    -o "${evidence}/context-${mode}"
  "${environment[@]}" "${evidence}/context-${mode}" \
    >"${evidence}/context-${mode}.stdout" \
    2>"${evidence}/context-${mode}.stderr"
  test ! -s "${evidence}/context-${mode}.stderr"
done
"${evidence}/context-normal" >"${evidence}/context-repeat.stdout" \
  2>"${evidence}/context-repeat.stderr"
test ! -s "${evidence}/context-repeat.stderr"
cmp "${evidence}/context-normal.stdout" "${evidence}/context-repeat.stdout"
cmp "${evidence}/context-normal.stdout" "${evidence}/context-sanitize.stdout"
grep -F 'G3-C4 context retention: registry_1024=1024 bytes_1024=49152 registry_8192=8192 bytes_8192=393216' \
  "${evidence}/context-normal.stdout" >/dev/null
grep -E '^G3-C4 context cache PASS: checks=[1-9][0-9]* record_bytes=48$' \
  "${evidence}/context-normal.stdout" >/dev/null
"${PROJECT_ROOT}/scripts/test-g3c4-native-owner.sh" \
  "${evidence}/owner-regression"
cat "${evidence}/context-normal.stdout"
printf 'G3-C4 context/cache evidence: %s\n' "${evidence}"
