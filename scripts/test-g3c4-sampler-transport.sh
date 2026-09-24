#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in cmp comm git nm python3 sha256sum; do
  require_command "${command}"
done
cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
evidence="${1:-$(project_build_dir)/g3c4-sampler-transport-test}"
mkdir -p "${evidence}"

python3 "${PROJECT_ROOT}/scripts/check-g3c4-sampler-transport.py" \
  >"${evidence}/static.stdout"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)
forward_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROVIDER_ROUTES_PRIVATE -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
)
sampler_macros=("${forward_macros[@]}" -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE)

"${cc}" "${flags[@]}" -O2 "${forward_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/full-prefix-owner.o"
"${cc}" "${flags[@]}" -O2 "${sampler_macros[@]}" -MMD \
  -MF "${evidence}/sampler-transport.d" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/sampler-transport-owner.o"
for object in full-prefix-owner sampler-transport-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/full-prefix-owner-defined.txt" \
  "${evidence}/sampler-transport-owner-defined.txt" \
  >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_sample_last_v1 |
  cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/full-prefix-owner-undefined.txt" \
  "${evidence}/sampler-transport-owner-undefined.txt" \
  >"${evidence}/added-undefined.txt"
test ! -s "${evidence}/added-undefined.txt"

if "${cc}" "${flags[@]}" -O2 \
    -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/invalid-sampler-only.o" \
    >"${evidence}/invalid-sampler-only.stdout" \
    2>"${evidence}/invalid-sampler-only.stderr"; then
  printf 'invalid sampler-only macro tuple compiled\n' >&2
  exit 1
fi
grep -F 'ET_G3C4_SAMPLER_TRANSPORT_PRIVATE requires full-prefix forward' \
  "${evidence}/invalid-sampler-only.stderr" >/dev/null

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_sampler_transport.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3c4_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3s_sampling_provider.c"
  "${PROJECT_ROOT}/native/a2_kv_cache.c"
)
for mode in normal sanitize; do
  mode_flags=(-O2)
  environment=()
  if [[ "${mode}" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "${cc}" "${flags[@]}" "${mode_flags[@]}" -DET_A2_KV_CACHE_TESTING \
    "${sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch -lm \
    -o "${evidence}/sampler-${mode}"
  "${environment[@]}" "${evidence}/sampler-${mode}" \
    >"${evidence}/sampler-${mode}.stdout" \
    2>"${evidence}/sampler-${mode}.stderr"
  test ! -s "${evidence}/sampler-${mode}.stderr"
done
"${evidence}/sampler-normal" >"${evidence}/sampler-repeat.stdout" \
  2>"${evidence}/sampler-repeat.stderr"
test ! -s "${evidence}/sampler-repeat.stderr"
cmp "${evidence}/sampler-normal.stdout" "${evidence}/sampler-repeat.stdout"
cmp "${evidence}/sampler-normal.stdout" "${evidence}/sampler-sanitize.stdout"
grep -E '^G3-C4 sampler transport PASS: checks=[1-9][0-9]* dispatches=1 modes=2$' \
  "${evidence}/sampler-normal.stdout" >/dev/null

git -C "${PROJECT_ROOT}" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_sampler_transport_source_closure.txt" \
  >"${evidence}/closure.sha256"
cat "${evidence}/static.stdout"
cat "${evidence}/sampler-normal.stdout"
printf 'G3-C4 private sampler evidence: %s\n' "${evidence}"
