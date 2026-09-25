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
evidence="${1:-$(project_build_dir)/g3c4-prefill1-test}"
mkdir -p "${evidence}"

python3 "${PROJECT_ROOT}/scripts/check-g3c4-prefill1.py" \
  >"${evidence}/static.stdout"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)
prefill3_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROVIDER_ROUTES_PRIVATE -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
  -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE -DET_G3C4_TOKEN_FRAME_PRIVATE
  -DET_G3C4_TOKEN_FORWARD_PRIVATE -DET_G3C4_PREFILL3_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
)
prefill1_macros=("${prefill3_macros[@]}" -DET_G3C4_PREFILL1_PRIVATE)

"${cc}" "${flags[@]}" -O2 "${prefill3_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/prefill3-owner.o"
"${cc}" "${flags[@]}" -O2 "${prefill1_macros[@]}" -MMD \
  -MF "${evidence}/prefill1.d" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/prefill1-owner.o"
for object in prefill3-owner prefill1-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/prefill3-owner-defined.txt" \
  "${evidence}/prefill1-owner-defined.txt" \
  >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_prefill1_v1 |
  cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/prefill3-owner-undefined.txt" \
  "${evidence}/prefill1-owner-undefined.txt" \
  >"${evidence}/added-undefined.txt"
test ! -s "${evidence}/added-undefined.txt"

if "${cc}" "${flags[@]}" -O2 \
    -DET_G3C4_PREFILL1_PRIVATE -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/invalid-prefill1-only.o" \
    >"${evidence}/invalid-prefill1-only.stdout" \
    2>"${evidence}/invalid-prefill1-only.stderr"; then
  printf 'invalid P1-prefill-only macro tuple compiled\n' >&2
  exit 1
fi
grep -F 'ET_G3C4_PREFILL1_PRIVATE requires Step 12A' \
  "${evidence}/invalid-prefill1-only.stderr" >/dev/null

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_prefill1.c"
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
    -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
    "${sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch -lm \
    -o "${evidence}/prefill1-${mode}"
  "${environment[@]}" "${evidence}/prefill1-${mode}" \
    >"${evidence}/prefill1-${mode}.stdout" \
    2>"${evidence}/prefill1-${mode}.stderr"
  test ! -s "${evidence}/prefill1-${mode}.stderr"
done
"${evidence}/prefill1-normal" \
  >"${evidence}/prefill1-repeat.stdout" \
  2>"${evidence}/prefill1-repeat.stderr"
test ! -s "${evidence}/prefill1-repeat.stderr"
cmp "${evidence}/prefill1-normal.stdout" \
  "${evidence}/prefill1-repeat.stdout"
cmp "${evidence}/prefill1-normal.stdout" \
  "${evidence}/prefill1-sanitize.stdout"
grep -E '^G3-C4 prefill1 PASS: checks=[1-9][0-9]* roles=21 dispatch-cuts=21 a2-allocation-cuts=11 runtime-allocation-cuts=2$' \
  "${evidence}/prefill1-normal.stdout" >/dev/null

git -C "${PROJECT_ROOT}" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_prefill1_source_closure.txt" \
  >"${evidence}/closure.sha256"
cat "${evidence}/static.stdout"
cat "${evidence}/prefill1-normal.stdout"
printf 'G3-C4 private P1 prefill evidence: %s\n' "${evidence}"
