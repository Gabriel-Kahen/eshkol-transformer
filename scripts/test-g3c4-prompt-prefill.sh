#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evidence="${1:-${PROJECT_ROOT}/build/g3c4-prompt-prefill-test}"
cc="${CC:-clang}"
mkdir -p "${evidence}"

for command in "${cc}" cmp comm git nm python3 sha256sum; do
  command -v "${command}" >/dev/null 2>&1 || {
    printf 'missing command: %s\n' "${command}" >&2
    exit 1
  }
done

python3 "${PROJECT_ROOT}/scripts/check-g3c4-prompt-prefill.py" \
  >"${evidence}/static.stdout"

flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -I "${PROJECT_ROOT}/src"
  -I "${PROJECT_ROOT}/src/eshkol_transformer"
)
base_macros=(
  -DET_G3C4_CONTEXT_PRIVATE -DET_G3C4_NATIVE_PINS_PRIVATE
  -DET_G3C4_ACTIVE_CALL_PRIVATE -DET_G3C4_GENERATOR_PRIVATE
  -DET_G3C4_PROMPT_T1_BORROW_PRIVATE
  -DET_G3C4_PROVIDER_ROUTES_PRIVATE -DET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
  -DET_G3C4_SAMPLER_TRANSPORT_PRIVATE -DET_G3C4_TOKEN_FRAME_PRIVATE
  -DET_G3C4_TOKEN_FORWARD_PRIVATE -DET_G3C4_PREFILL3_PRIVATE
  -DET_G3C4_PREFILL1_PRIVATE -DET_G3C4_PREFILL2_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
)
binding_macros=("${base_macros[@]}" -DET_G3C4_PROMPT_PREFILL_PRIVATE)

"${cc}" "${flags[@]}" -O2 "${base_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/base-owner.o"
"${cc}" "${flags[@]}" -O2 "${binding_macros[@]}" -MMD \
  -MF "${evidence}/binding.d" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/binding-owner.o"
for object in base-owner binding-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/base-owner-defined.txt" \
  "${evidence}/binding-owner-defined.txt" \
  >"${evidence}/added-defined.txt"
printf '%s\n' \
  et_g3c4_private_prompt_prefill_preflight_v1 \
  et_g3c4_private_prompt_prefill_v1 |
  LC_ALL=C sort | cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/base-owner-undefined.txt" \
  "${evidence}/binding-owner-undefined.txt" \
  >"${evidence}/added-undefined.txt"
printf '%s\n' \
  et_i64_tensor_borrow_begin_v1 \
  et_i64_tensor_borrow_end_v1 \
  et_i64_tensor_borrow_view_v1 |
  LC_ALL=C sort | cmp - "${evidence}/added-undefined.txt"

if "${cc}" "${flags[@]}" -O2 -DET_G3C4_PROMPT_PREFILL_PRIVATE -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/invalid-binding-only.o" \
    >"${evidence}/invalid-binding-only.stdout" \
    2>"${evidence}/invalid-binding-only.stderr"; then
  printf 'invalid binding-only macro tuple compiled\n' >&2
  exit 1
fi
grep -F 'ET_G3C4_PROMPT_PREFILL_PRIVATE requires Steps 15A and 17A' \
  "${evidence}/invalid-binding-only.stderr" >/dev/null

sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_prompt_prefill.c"
  "${PROJECT_ROOT}/src/eshkol_transformer/m3_i64_integration.c"
  "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c"
  "${PROJECT_ROOT}/native/t1_i64_shell.c"
  "${PROJECT_ROOT}/native/n3k_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3c4_primitives_provider.c"
  "${PROJECT_ROOT}/native/g3s_sampling_provider.c"
  "${PROJECT_ROOT}/native/g3n_primitives_provider.c"
  "${PROJECT_ROOT}/native/n2_primitives_provider.c"
  "${PROJECT_ROOT}/native/a2_kv_cache.c"
)

compile_mode() {
  local mode=$1
  local -a mode_flags=(-O2) environment=()
  if [[ "${mode}" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    environment=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1)
  fi
  "${cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_I64_TENSOR_TESTING -DET_A2_KV_CACHE_TESTING \
    -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
    "${sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch -lm \
    -o "${evidence}/prompt-prefill-${mode}"
  "${environment[@]}" "${evidence}/prompt-prefill-${mode}" \
    >"${evidence}/${mode}.stdout" 2>"${evidence}/${mode}.stderr"
  test ! -s "${evidence}/${mode}.stderr"
}

compile_mode normal
"${evidence}/prompt-prefill-normal" \
  >"${evidence}/runtime-repeat.stdout" \
  2>"${evidence}/runtime-repeat.stderr"
test ! -s "${evidence}/runtime-repeat.stderr"
compile_mode sanitize
cmp "${evidence}/normal.stdout" "${evidence}/runtime-repeat.stdout"
cmp "${evidence}/normal.stdout" "${evidence}/sanitize.stdout"
grep -E '^G3-C4 prompt/prefill binding PASS: checks=[1-9][0-9]* routes=2 pre-acquire-cuts=5 numerical-cuts=42 borrow-cuts=3$' \
  "${evidence}/normal.stdout" >/dev/null

git -C "${PROJECT_ROOT}" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_prompt_prefill_source_closure.txt" \
  >"${evidence}/closure.sha256"
cat "${evidence}/static.stdout"
cat "${evidence}/normal.stdout"
printf 'G3-C4 private prompt/prefill evidence: %s\n' "${evidence}"
