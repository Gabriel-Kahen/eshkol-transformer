#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp comm git nm python3 tar; do require_command "${command}"; done
generator_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
generator_cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
evidence="${1:-$(project_build_dir)/g3c4-generator-test}"
mkdir -p "${evidence}/base"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)

python3 "${PROJECT_ROOT}/scripts/check-g3c4-generator.py"
git -C "${PROJECT_ROOT}" archive \
  7d774d0255c589b0b65cdf47e5413ae5d0348ecb |
  tar -x -C "${evidence}/base"

identity_names=(ordinary context active)
identity_flags=(
  ''
  '-DET_G3C4_CONTEXT_PRIVATE'
  '-DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE -DET_G3C4_ACTIVE_CALL_PRIVATE'
)
for index in 0 1 2; do
  read -r -a mode_flags <<<"${identity_flags[index]}"
  "${generator_cc}" "${flags[@]}" -O2 "${mode_flags[@]}" -c \
    "${evidence}/base/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/base-${identity_names[index]}.o"
  "${generator_cc}" "${flags[@]}" -O2 "${mode_flags[@]}" -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/current-${identity_names[index]}.o"
  cmp "${evidence}/base-${identity_names[index]}.o" \
    "${evidence}/current-${identity_names[index]}.o"
done

generator_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
)
"${generator_cc}" "${flags[@]}" -O2 "${generator_macros[@]}" -c \
  -MMD -MF "${evidence}/generator-owner.d" -MT generator-owner.o \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/generator-owner.o"
sed "s#${PROJECT_ROOT}/##g" "${evidence}/generator-owner.d" \
  >"${evidence}/generator-owner-normalized.d"
cmp "${PROJECT_ROOT}/native/g3c4_generator_owner_deps.txt" \
  "${evidence}/generator-owner-normalized.d"
for object in current-active generator-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/current-active-defined.txt" \
  "${evidence}/generator-owner-defined.txt" >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_generator_close_v1 \
  et_g3c4_private_generator_rng_v1 et_g3c4_private_generator_seed_v1 \
  et_g3c4_private_rng_clone_v1 et_g3c4_private_rng_release_v1 \
  et_g3c4_private_rng_seed_v1 et_g3c4_private_rng_word_v1 |
  LC_ALL=C sort | cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/current-active-undefined.txt" \
  "${evidence}/generator-owner-undefined.txt" \
  >"${evidence}/added-undefined.txt"
test ! -s "${evidence}/added-undefined.txt"

invalid_names=(
  generator_only generator_context generator_pins generator_active
  generator_context_pins generator_context_active generator_pins_active
)
invalid_flags=(
  '-DET_G3C4_GENERATOR_PRIVATE'
  '-DET_G3C4_GENERATOR_PRIVATE -DET_G3C4_CONTEXT_PRIVATE'
  '-DET_G3C4_GENERATOR_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE'
  '-DET_G3C4_GENERATOR_PRIVATE -DET_G3C4_ACTIVE_CALL_PRIVATE'
  '-DET_G3C4_GENERATOR_PRIVATE -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE'
  '-DET_G3C4_GENERATOR_PRIVATE -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_ACTIVE_CALL_PRIVATE'
  '-DET_G3C4_GENERATOR_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE -DET_G3C4_ACTIVE_CALL_PRIVATE'
)
for index in 0 1 2 3 4 5 6; do
  read -r -a tuple_flags <<<"${invalid_flags[index]}"
  if "${generator_cc}" "${flags[@]}" -O2 "${tuple_flags[@]}" -c \
      "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
      -o "${evidence}/${invalid_names[index]}.o" \
      >"${evidence}/${invalid_names[index]}.stdout" \
      2>"${evidence}/${invalid_names[index]}.stderr"; then
    printf 'invalid macro tuple compiled: %s\n' "${invalid_names[index]}" >&2
    exit 1
  fi
  grep -F 'ET_G3C4_GENERATOR_PRIVATE requires context, C4 pins, and active call' \
    "${evidence}/${invalid_names[index]}.stderr" >/dev/null
done

cat >"${evidence}/header-c.c" <<'EOF'
#define ET_G3C4_GENERATOR_PRIVATE 1
#include "eshkol_transformer/g3c4_context_internal.h"
int main(void) { return 0; }
EOF
cat >"${evidence}/header-cpp.cpp" <<'EOF'
#define ET_G3C4_GENERATOR_PRIVATE 1
#include "eshkol_transformer/g3c4_context_internal.h"
int main() { return 0; }
EOF
"${generator_cc}" "${flags[@]}" -c "${evidence}/header-c.c" \
  -o "${evidence}/header-c.o"
"${generator_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/src" -c "${evidence}/header-cpp.cpp" \
  -o "${evidence}/header-cpp.o"

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_generator_native.c"
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
  "${generator_cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING -DET_G3C4_GENERATOR_A2_TU \
    -c "${PROJECT_ROOT}/tests/g3c4/test_generator_native.c" \
    -o "${evidence}/generator-a2-${mode}.o"
  "${generator_cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING "${sources[@]}" \
    "${evidence}/generator-a2-${mode}.o" -lm \
    -o "${evidence}/generator-${mode}"
  "${environment[@]}" "${evidence}/generator-${mode}" \
    >"${evidence}/generator-${mode}.stdout" \
    2>"${evidence}/generator-${mode}.stderr"
  test ! -s "${evidence}/generator-${mode}.stderr"
done
"${evidence}/generator-normal" >"${evidence}/generator-repeat.stdout" \
  2>"${evidence}/generator-repeat.stderr"
test ! -s "${evidence}/generator-repeat.stderr"
cmp "${evidence}/generator-normal.stdout" \
  "${evidence}/generator-repeat.stdout"
cmp "${evidence}/generator-normal.stdout" \
  "${evidence}/generator-sanitize.stdout"
grep -F 'G3-C4 generator retention: contexts_1024=1024 bytes_1024=1556480 contexts_8192=8192 bytes_8192=12451840 rng_1024=1024 rng_bytes_1024=65536 rng_8192=8192 rng_bytes_8192=524288' \
  "${evidence}/generator-normal.stdout" >/dev/null
grep -E '^G3-C4 native generator PASS: checks=[1-9][0-9]* context_bytes=1520 rng_bytes=64$' \
  "${evidence}/generator-normal.stdout" >/dev/null

"${PROJECT_ROOT}/scripts/test-g3c4-active-call.sh" \
  "${evidence}/active-regression" >/dev/null
cat "${evidence}/generator-normal.stdout"
printf 'G3-C4 native generator evidence: %s\n' "${evidence}"
