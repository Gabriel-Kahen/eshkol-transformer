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
evidence="${1:-$(project_build_dir)/g3c4-prefill3-test}"
mkdir -p "${evidence}"

python3 "${PROJECT_ROOT}/scripts/check-g3c4-prefill3.py" \
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
  -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE -DET_G3C4_TOKEN_FRAME_PRIVATE
  -DET_G3C4_TOKEN_FORWARD_PRIVATE
)
prefill_macros=("${forward_macros[@]}" -DET_G3C4_PREFILL3_PRIVATE)

"${cc}" "${flags[@]}" -O2 "${forward_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/token-forward-owner.o"
"${cc}" "${flags[@]}" -O2 "${prefill_macros[@]}" -MMD \
  -MF "${evidence}/prefill3.d" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/prefill3-owner.o"
for object in token-forward-owner prefill3-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/token-forward-owner-defined.txt" \
  "${evidence}/prefill3-owner-defined.txt" \
  >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_prefill3_v1 |
  cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/token-forward-owner-undefined.txt" \
  "${evidence}/prefill3-owner-undefined.txt" \
  >"${evidence}/added-undefined.txt"
printf '%s\n' bcmp | cmp - "${evidence}/added-undefined.txt"

if "${cc}" "${flags[@]}" -O2 \
    -DET_G3C4_PREFILL3_PRIVATE -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/invalid-prefill-only.o" \
    >"${evidence}/invalid-prefill-only.stdout" \
    2>"${evidence}/invalid-prefill-only.stderr"; then
  printf 'invalid prefill-only macro tuple compiled\n' >&2
  exit 1
fi
grep -F 'ET_G3C4_PREFILL3_PRIVATE requires Step 11A' \
  "${evidence}/invalid-prefill-only.stderr" >/dev/null

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_prefill3.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3c4_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3s_sampling_provider.c"
  "${PROJECT_ROOT}/native/g3n_primitives_provider.c"
  "${PROJECT_ROOT}/native/n2_primitives_provider.c"
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
    -o "${evidence}/prefill3-${mode}"
  "${environment[@]}" "${evidence}/prefill3-${mode}" \
    >"${evidence}/prefill3-${mode}.stdout" \
    2>"${evidence}/prefill3-${mode}.stderr"
  test ! -s "${evidence}/prefill3-${mode}.stderr"
done
"${evidence}/prefill3-normal" \
  >"${evidence}/prefill3-repeat.stdout" \
  2>"${evidence}/prefill3-repeat.stderr"
test ! -s "${evidence}/prefill3-repeat.stderr"
cmp "${evidence}/prefill3-normal.stdout" \
  "${evidence}/prefill3-repeat.stdout"
cmp "${evidence}/prefill3-normal.stdout" \
  "${evidence}/prefill3-sanitize.stdout"
grep -E '^G3-C4 prefill3 PASS: checks=[1-9][0-9]* roles=21 dispatch-cuts=21 allocation-cuts=14$' \
  "${evidence}/prefill3-normal.stdout" >/dev/null

git -C "${PROJECT_ROOT}" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_prefill3_source_closure.txt" \
  >"${evidence}/closure.sha256"
cat "${evidence}/static.stdout"
cat "${evidence}/prefill3-normal.stdout"
printf 'G3-C4 private prefill3 evidence: %s\n' "${evidence}"
