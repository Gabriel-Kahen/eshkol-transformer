#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp comm env git grep nm python3 sha256sum timeout; do
  require_command "${command}"
done

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
eshkol_runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-output-text-test}"
mkdir -p "${evidence}"

python3 "${PROJECT_ROOT}/scripts/check-g3c4-output-text.py" \
  >"${evidence}/static.stdout"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.q0.test_python_isolation \
  >"${evidence}/python-isolation.stdout" 2>"${evidence}/python-isolation.stderr"

if [[ "${G3C4_SKIP_OUTPUT_DECODE_IDS_REGRESSION:-0}" == 1 ]]; then
  printf 'G3-C4 output decode-ID regression: SKIPPED\n' \
    >"${evidence}/output-decode-ids-regression.stdout"
else
  CC="${cc}" "${PROJECT_ROOT}/scripts/test-g3c4-output-decode-ids.sh" \
    "${evidence}/output-decode-ids-regression" \
    >"${evidence}/output-decode-ids-regression.stdout" 2>&1
fi

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
  -DET_G3C4_PROMPT_PREFILL_PRIVATE
  -DET_G3C4_OUTPUT_RESERVATION_PRIVATE
  -DET_G3C4_LAST_LOGIT_FRAME_PRIVATE
  -DET_G3C4_OUTPUT_PREPARE_PRIVATE
  -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_G3C4_OUTPUT_DECODE_IDS_PRIVATE
)
text_macros=("${base_macros[@]}" -DET_G3C4_OUTPUT_TEXT_PRIVATE)

"${cc}" "${flags[@]}" -O2 "${base_macros[@]}" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/base-owner.o"
"${cc}" "${flags[@]}" -O2 "${text_macros[@]}" -MMD \
  -MF "${evidence}/text.d" -c \
  "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
  -o "${evidence}/text-owner.o"
for object in base-owner text-owner; do
  nm -g --defined-only --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 2 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-defined.txt"
  nm -u --format=posix "${evidence}/${object}.o" |
    awk 'NF >= 1 { print $1 }' | LC_ALL=C sort -u \
    >"${evidence}/${object}-undefined.txt"
done
comm -13 "${evidence}/base-owner-defined.txt" \
  "${evidence}/text-owner-defined.txt" >"${evidence}/added-defined.txt"
printf '%s\n' et_g3c4_private_output_accept_text_v1 |
  LC_ALL=C sort | cmp - "${evidence}/added-defined.txt"
comm -13 "${evidence}/base-owner-undefined.txt" \
  "${evidence}/text-owner-undefined.txt" >"${evidence}/added-undefined.txt"
test ! -s "${evidence}/added-undefined.txt"

if "${cc}" "${flags[@]}" -O2 -DET_G3C4_OUTPUT_TEXT_PRIVATE -c \
    "${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c" \
    -o "${evidence}/invalid-text-only.o" \
    >"${evidence}/invalid-text-only.stdout" \
    2>"${evidence}/invalid-text-only.stderr"; then
  printf 'invalid text-only macro tuple compiled\n' >&2
  exit 1
fi
grep -F 'ET_G3C4_OUTPUT_TEXT_PRIVATE requires Step 21A output ID staging' \
  "${evidence}/invalid-text-only.stderr" >/dev/null

native_sources=(
  "${PROJECT_ROOT}/tests/g3c4/test_output_text.c"
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

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-output-text.XXXXXX")"
cleanup() {
  local status=$?
  if (( status == 0 )); then
    rm -rf -- "${tmp}"
  else
    printf 'G3-C4 output text failure evidence retained at %s\n' "${tmp}" >&2
    find "${tmp}" -type f \
      \( -name '*.stderr' -o -name '*.stdout' \) \
      -print -exec tail -200 {} \; >&2 || true
  fi
}
trap cleanup EXIT

compile_mode() {
  local mode=$1 directory="${tmp}/$1"
  local -a mode_flags=(-O2) runtime=(env)
  mkdir -p "${directory}/objects" "${directory}/cache"
  if [[ "${mode}" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi

  "${cc}" "${flags[@]}" "${mode_flags[@]}" \
    -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE \
    -DET_A2_KV_CACHE_TESTING -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
    "${native_sources[@]}" -Wl,--wrap=et_kernel_runtime_dispatch -lm \
    -o "${directory}/output-text"
  "${runtime[@]}" "${directory}/output-text" \
    >"${directory}/native.stdout" 2>"${directory}/native.stderr"
  test ! -s "${directory}/native.stderr"
  grep -E '^G3-C4 output text PASS: checks=[1-9][0-9]* routes=2 readiness-cuts=3 malformed-cuts=4 alias-cuts=7 borrow-cuts=2 ownership-cuts=2 mutation-cuts=1$' \
    "${directory}/native.stdout" >/dev/null

  local -a common_flags=(
    -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
    -fvisibility=hidden -fno-common -fstack-protector-all
    -ffp-contract=off -fexcess-precision=standard -fno-fast-math
    -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
    -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer"
    -I "$(eshkol_source_dir)/inc"
  )
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -DET_I2_NATIVE_HELPERS_ONLY -DET_G3C4_I2_CONSTRUCTION_PRIVATE \
    -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    -o "${directory}/objects/i2_native_helpers.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -DET_G3C4_NATIVE_OWNER_PRIVATE \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" \
    -o "${directory}/objects/m3t_f32_integration.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -c "${PROJECT_ROOT}/tests/g3c4/prompt_t1_borrow_native.c" \
    -o "${directory}/objects/g3c4_prompt_t1_borrow_native.o"
  "${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -fstack-protector-all \
    "${mode_flags[@]}" -I "$(eshkol_source_dir)/inc" \
    -I "$(eshkol_source_dir)/lib/core" \
    -c "${PROJECT_ROOT}/tests/g3c4/call_entry_allocation_shim.cpp" \
    -o "${directory}/objects/g3c4_call_entry_allocation_shim.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -DET_A2_KV_CACHE_TESTING \
    -c "${PROJECT_ROOT}/tests/g3c4/call_entry_a2_stats.c" \
    -o "${directory}/objects/g3c4_call_entry_a2_stats.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -DET_P1_TRUSTED_BUILD=1 -c "${PROJECT_ROOT}/native/p1_identity.c" \
    -o "${directory}/objects/p1_identity.o"
  local source
  for source in data_io checkpoint_io kernel_abi \
                n3k_primitives_provider n2_primitives_provider \
                a2_attention_provider; do
    "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
      -DET_A2_KV_CACHE_TESTING -c "${PROJECT_ROOT}/native/${source}.c" \
      -o "${directory}/objects/${source}.o"
  done
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -DET_T1_I64_SHELL_TESTING -c "${PROJECT_ROOT}/native/t1_i64_shell.c" \
    -o "${directory}/objects/t1_i64_shell.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -DET_I64_TENSOR_TESTING -DET_M3_TESTING \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/m3_i64_integration.c" \
    -o "${directory}/objects/m3_i64_integration.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/m3_model.c" \
    -o "${directory}/objects/m3_model.o"
  ar rcsD "${directory}/libg3c4_output_text.a" \
    "${directory}"/objects/*.o

  local sanitize_link=
  if [[ "${mode}" == sanitize ]]; then
    sanitize_link='-fsanitize=address,undefined -fno-omit-frame-pointer'
  fi
  cat >"${directory}/cxx-wrap" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") ${sanitize_link} "\$@" \
  -Wl,--wrap=arena_allocate_vector_with_header \
  -Wl,--wrap=arena_allocate_cons_with_header \
  -Wl,--wrap=malloc -Wl,--wrap=eshkol_push_exception_handler
WRAPPER
  chmod 0500 "${directory}/cxx-wrap"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${directory}/cache" \
    ESHKOL_CXX_COMPILER="${directory}/cxx-wrap" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s \
      "${eshkol_runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      -L "${directory}" --lib g3c4_output_text \
      "${PROJECT_ROOT}/tests/g3c4/t1_output_decode_test.esk" \
      -o "${directory}/t1-output-decode" \
      >"${directory}/compile.stdout" 2>"${directory}/compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${directory}/t1-output-decode" \
    >"${directory}/eshkol.stdout" 2>"${directory}/eshkol.stderr"
  test ! -s "${directory}/eshkol.stderr"
  grep -E '^G3-C4 T1 output decode PASS: checks=[1-9][0-9]* routes=2 allocation-free=2 mutation-cuts=1$' \
    "${directory}/eshkol.stdout" >/dev/null
}

compile_mode normal
"${tmp}/normal/output-text" >"${tmp}/normal/runtime-repeat.stdout" \
  2>"${tmp}/normal/runtime-repeat.stderr"
test ! -s "${tmp}/normal/runtime-repeat.stderr"
ESHKOL_ARENA_POISON=1 "${tmp}/normal/t1-output-decode" \
  >"${tmp}/normal/eshkol-repeat.stdout" \
  2>"${tmp}/normal/eshkol-repeat.stderr"
test ! -s "${tmp}/normal/eshkol-repeat.stderr"
compile_mode sanitize
cmp "${tmp}/normal/native.stdout" "${tmp}/normal/runtime-repeat.stdout"
cmp "${tmp}/normal/native.stdout" "${tmp}/sanitize/native.stdout"
cmp "${tmp}/normal/eshkol.stdout" "${tmp}/normal/eshkol-repeat.stdout"
cmp "${tmp}/normal/eshkol.stdout" "${tmp}/sanitize/eshkol.stdout"

for mode in normal sanitize; do
  cp "${tmp}/${mode}/native.stdout" "${evidence}/${mode}-native.stdout"
  cp "${tmp}/${mode}/native.stderr" "${evidence}/${mode}-native.stderr"
  cp "${tmp}/${mode}/compile.stdout" "${evidence}/${mode}-compile.stdout"
  cp "${tmp}/${mode}/compile.stderr" "${evidence}/${mode}-compile.stderr"
  cp "${tmp}/${mode}/eshkol.stdout" "${evidence}/${mode}-eshkol.stdout"
  cp "${tmp}/${mode}/eshkol.stderr" "${evidence}/${mode}-eshkol.stderr"
done
cp "${tmp}/normal/runtime-repeat.stdout" "${evidence}/runtime-repeat.stdout"
cp "${tmp}/normal/eshkol-repeat.stdout" "${evidence}/eshkol-repeat.stdout"
git -C "${PROJECT_ROOT}" diff --check
sha256sum "${PROJECT_ROOT}/native/g3c4_output_text_source_closure.txt" \
  >"${evidence}/closure.sha256"
cat "${evidence}/static.stdout"
cat "${evidence}/normal-native.stdout"
cat "${evidence}/normal-eshkol.stdout"
printf 'G3-C4 private output text evidence: %s\n' "${evidence}"
