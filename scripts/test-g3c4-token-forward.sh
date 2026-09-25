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
evidence="${1:-$(project_build_dir)/g3c4-token-forward-test}"
mkdir -p "${evidence}"

python3 "${PROJECT_ROOT}/scripts/check-g3c4-token-forward.py" \
  >"${evidence}/static.stdout"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)
frame_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROVIDER_ROUTES_PRIVATE -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
  -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE -DET_G3C4_TOKEN_FRAME_PRIVATE
)
forward_macros=("${frame_macros[@]}" -DET_G3C4_TOKEN_FORWARD_PRIVATE)

"${cc}" "${flags[@]}" -O2 "${frame_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/token-frame-owner.o"
"${cc}" "${flags[@]}" -O2 "${forward_macros[@]}" -MMD \
  -MF "${evidence}/token-forward.d" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/token-forward-owner.o"
for object in token-frame-owner token-forward-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/token-frame-owner-defined.txt" \
  "${evidence}/token-forward-owner-defined.txt" \
  >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_token_forward_v1 |
  cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/token-frame-owner-undefined.txt" \
  "${evidence}/token-forward-owner-undefined.txt" \
  >"${evidence}/added-undefined.txt"
printf '%s\n' \
  et_a2_kv_cache_read_borrow_layer_v1 \
  et_g3n_kernel_provider_v1 \
  et_n2_kernel_provider_v1 | LC_ALL=C sort |
  cmp - "${evidence}/added-undefined.txt"

if "${cc}" "${flags[@]}" -O2 \
    -DET_G3C4_TOKEN_FORWARD_PRIVATE -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/invalid-forward-only.o" \
    >"${evidence}/invalid-forward-only.stdout" \
    2>"${evidence}/invalid-forward-only.stderr"; then
  printf 'invalid token-forward-only macro tuple compiled\n' >&2
  exit 1
fi
grep -F 'ET_G3C4_TOKEN_FORWARD_PRIVATE requires Step 10A' \
  "${evidence}/invalid-forward-only.stderr" >/dev/null

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_token_forward.c"
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
    -o "${evidence}/token-forward-${mode}"
  "${environment[@]}" "${evidence}/token-forward-${mode}" \
    >"${evidence}/token-forward-${mode}.stdout" \
    2>"${evidence}/token-forward-${mode}.stderr"
  test ! -s "${evidence}/token-forward-${mode}.stderr"
done
"${evidence}/token-forward-normal" \
  >"${evidence}/token-forward-repeat.stdout" \
  2>"${evidence}/token-forward-repeat.stderr"
test ! -s "${evidence}/token-forward-repeat.stderr"
cmp "${evidence}/token-forward-normal.stdout" \
  "${evidence}/token-forward-repeat.stdout"
cmp "${evidence}/token-forward-normal.stdout" \
  "${evidence}/token-forward-sanitize.stdout"
grep -E '^G3-C4 token forward PASS: checks=[1-9][0-9]* roles=21 cuts=31 positions=2$' \
  "${evidence}/token-forward-normal.stdout" >/dev/null

git -C "${PROJECT_ROOT}" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_token_forward_source_closure.txt" \
  >"${evidence}/closure.sha256"
cat "${evidence}/static.stdout"
cat "${evidence}/token-forward-normal.stdout"
printf 'G3-C4 private token-forward evidence: %s\n' "${evidence}"
