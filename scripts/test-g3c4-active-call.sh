#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp comm git nm python3 tar; do require_command "${command}"; done
active_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
active_cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
evidence="${1:-$(project_build_dir)/g3c4-active-call-test}"
mkdir -p "${evidence}/base"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)

python3 "${PROJECT_ROOT}/scripts/check-g3c4-active-call.py"
git -C "${PROJECT_ROOT}" archive \
  58c59699365cce8d0ef3dac1a8141dc481712ec9 |
  tar -x -C "${evidence}/base"

for mode in ordinary context; do
  mode_flags=()
  if [[ "${mode}" == context ]]; then
    mode_flags=(-DET_G3C4_CONTEXT_PRIVATE)
  fi
  "${active_cc}" "${flags[@]}" -O2 "${mode_flags[@]}" -c \
    "${evidence}/base/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/base-${mode}.o"
  "${active_cc}" "${flags[@]}" -O2 "${mode_flags[@]}" -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/current-${mode}.o"
  cmp "${evidence}/base-${mode}.o" "${evidence}/current-${mode}.o"
done

active_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE
)
"${active_cc}" "${flags[@]}" -O2 "${active_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/active-owner.o"
for object in current-context active-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/current-context-defined.txt" \
  "${evidence}/active-owner-defined.txt" >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_call_abort_v1 \
  et_g3c4_private_call_acquire_v1 et_g3c4_private_call_finish_v1 \
  et_g3c4_private_call_prepare_end_v1 | LC_ALL=C sort |
  cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/current-context-undefined.txt" \
  "${evidence}/active-owner-undefined.txt" >"${evidence}/added-undefined.txt"
printf '%s\n' et_a2_kv_cache_read_borrow_begin_v1 \
  et_a2_kv_cache_read_borrow_end_v1 et_g3c4_model_pins_begin_internal \
  et_g3c4_model_pins_check_internal et_g3c4_model_pins_end_internal |
  LC_ALL=C sort | cmp - "${evidence}/added-undefined.txt"

invalid_tuples=(
  active_only
  active_context
  active_pins
)
invalid_flags=(
  '-DET_G3C4_ACTIVE_CALL_PRIVATE'
  '-DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_CONTEXT_PRIVATE'
  '-DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE'
)
for index in 0 1 2; do
  read -r -a tuple_flags <<<"${invalid_flags[index]}"
  if "${active_cc}" "${flags[@]}" -O2 "${tuple_flags[@]}" -c \
      "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
      -o "${evidence}/${invalid_tuples[index]}.o" \
      >"${evidence}/${invalid_tuples[index]}.stdout" \
      2>"${evidence}/${invalid_tuples[index]}.stderr"; then
    printf 'invalid macro tuple compiled: %s\n' "${invalid_tuples[index]}" >&2
    exit 1
  fi
  grep -F 'ET_G3C4_ACTIVE_CALL_PRIVATE requires context and C4 pins' \
    "${evidence}/${invalid_tuples[index]}.stderr" >/dev/null
done

cat >"${evidence}/header-c.c" <<'EOF'
#include "eshkol_transformer/g3c4_context_internal.h"
int main(void) { return 0; }
EOF
cat >"${evidence}/header-cpp.cpp" <<'EOF'
#include "eshkol_transformer/g3c4_context_internal.h"
int main() { return 0; }
EOF
"${active_cc}" "${flags[@]}" -c "${evidence}/header-c.c" \
  -o "${evidence}/header-c.o"
"${active_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/src" -c "${evidence}/header-cpp.cpp" \
  -o "${evidence}/header-cpp.o"

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_active_call.c"
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
  "${active_cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_G3C4_ACTIVE_CALL_A2_TU \
    -c "${PROJECT_ROOT}/tests/g3c4/test_active_call.c" \
    -o "${evidence}/active-a2-${mode}.o"
  "${active_cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING "${sources[@]}" \
    "${evidence}/active-a2-${mode}.o" -lm \
    -o "${evidence}/active-${mode}"
  "${environment[@]}" "${evidence}/active-${mode}" \
    >"${evidence}/active-${mode}.stdout" \
    2>"${evidence}/active-${mode}.stderr"
  test ! -s "${evidence}/active-${mode}.stderr"
done
"${evidence}/active-normal" >"${evidence}/active-repeat.stdout" \
  2>"${evidence}/active-repeat.stderr"
test ! -s "${evidence}/active-repeat.stderr"
cmp "${evidence}/active-normal.stdout" "${evidence}/active-repeat.stdout"
cmp "${evidence}/active-normal.stdout" "${evidence}/active-sanitize.stdout"
grep -F 'G3-C4 active retention: registry_1024=1024 bytes_1024=1466368 registry_8192=8192 bytes_8192=11730944' \
  "${evidence}/active-normal.stdout" >/dev/null
grep -E '^G3-C4 active call PASS: checks=[1-9][0-9]* record_bytes=1432$' \
  "${evidence}/active-normal.stdout" >/dev/null

"${PROJECT_ROOT}/scripts/test-g3c4-context-cache.sh" \
  "${evidence}/context-regression"
"${PROJECT_ROOT}/scripts/test-g3c4-pins.sh" \
  "${evidence}/pins-regression"
cat "${evidence}/active-normal.stdout"
printf 'G3-C4 active-call evidence: %s\n' "${evidence}"
