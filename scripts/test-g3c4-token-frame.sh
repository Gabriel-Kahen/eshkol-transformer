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
evidence="${1:-$(project_build_dir)/g3c4-token-frame-test}"
mkdir -p "${evidence}"

python3 "${PROJECT_ROOT}/scripts/check-g3c4-token-frame.py" \
  >"${evidence}/static.stdout"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)
sampler_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROVIDER_ROUTES_PRIVATE -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
  -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE
)
frame_macros=("${sampler_macros[@]}" -DET_G3C4_TOKEN_FRAME_PRIVATE)

"${cc}" "${flags[@]}" -O2 "${sampler_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/sampler-owner.o"
"${cc}" "${flags[@]}" -O2 "${frame_macros[@]}" -MMD \
  -MF "${evidence}/token-frame.d" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/token-frame-owner.o"
for object in sampler-owner token-frame-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/sampler-owner-defined.txt" \
  "${evidence}/token-frame-owner-defined.txt" \
  >"${evidence}/added-defined.txt"
printf '%s\n' \
  et_g3c4_private_token_frame_abort_v1 \
  et_g3c4_private_token_frame_begin_v1 \
  et_g3c4_private_token_frame_publish_v1 \
  et_g3c4_private_token_frame_stage_v1 |
  LC_ALL=C sort | cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/sampler-owner-undefined.txt" \
  "${evidence}/token-frame-owner-undefined.txt" \
  >"${evidence}/added-undefined.txt"
printf '%s\n' et_a2_kv_cache_transaction_commit_v1 |
  cmp - "${evidence}/added-undefined.txt"

if "${cc}" "${flags[@]}" -O2 \
    -DET_G3C4_TOKEN_FRAME_PRIVATE -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/invalid-frame-only.o" \
    >"${evidence}/invalid-frame-only.stdout" \
    2>"${evidence}/invalid-frame-only.stderr"; then
  printf 'invalid token-frame-only macro tuple compiled\n' >&2
  exit 1
fi
grep -F 'ET_G3C4_TOKEN_FRAME_PRIVATE requires Step 9A' \
  "${evidence}/invalid-frame-only.stderr" >/dev/null

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_token_frame_commit.c"
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
    -o "${evidence}/token-frame-${mode}"
  "${environment[@]}" "${evidence}/token-frame-${mode}" \
    >"${evidence}/token-frame-${mode}.stdout" \
    2>"${evidence}/token-frame-${mode}.stderr"
  test ! -s "${evidence}/token-frame-${mode}.stderr"
done
"${evidence}/token-frame-normal" \
  >"${evidence}/token-frame-repeat.stdout" \
  2>"${evidence}/token-frame-repeat.stderr"
test ! -s "${evidence}/token-frame-repeat.stderr"
cmp "${evidence}/token-frame-normal.stdout" \
  "${evidence}/token-frame-repeat.stdout"
cmp "${evidence}/token-frame-normal.stdout" \
  "${evidence}/token-frame-sanitize.stdout"
grep -E '^G3-C4 token frame commit PASS: checks=[1-9][0-9]* sampler-dispatches=6 modes=2 cuts=4$' \
  "${evidence}/token-frame-normal.stdout" >/dev/null

git -C "${PROJECT_ROOT}" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_token_frame_source_closure.txt" \
  >"${evidence}/closure.sha256"
cat "${evidence}/static.stdout"
cat "${evidence}/token-frame-normal.stdout"
printf 'G3-C4 private token-frame evidence: %s\n' "${evidence}"
