#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp env grep python3 sha256sum timeout; do
  require_command "${command}"
done
python3 "${PROJECT_ROOT}/scripts/check-g3c4-prompt-t1-borrow.py"

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
eshkol_runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-prompt-t1-borrow-test}"
mkdir -p "${evidence}"

if [[ "${G3C4_SKIP_T1_EVAL_REGRESSION:-0}" == 1 ]]; then
  printf 'G3-C4 T1/eval regression: SKIPPED\n' \
    >"${evidence}/t1-eval-regression.stdout"
else
  "${PROJECT_ROOT}/scripts/test-g3c4-t1-eval-admission.sh" \
    "${evidence}/t1-eval-regression" \
    >"${evidence}/t1-eval-regression.stdout" 2>&1
fi

common_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer"
  -I "$(eshkol_source_dir)/inc"
)

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-prompt-t1.XXXXXX")"
cleanup() {
  local status=$?
  if (( status == 0 )); then
    rm -rf -- "${tmp}"
  else
    printf 'G3-C4 prompt/T1 failure evidence retained at %s\n' "${tmp}" >&2
    find "${tmp}" -type f \
      \( -name 'compile.stderr' -o -name 'runtime.stderr' -o -name 'runtime.stdout' \) \
      -print -exec cat {} \; >&2 || true
  fi
}
trap cleanup EXIT

compile_mode() {
  local mode=$1 directory="${tmp}/$1"
  local -a mode_flags=(-O2) runtime=(env)
  mkdir -p "${directory}/objects" "${directory}/cache" \
    "${directory}/retention-cache" "${directory}/allocation-cache"
  if [[ "${mode}" == sanitize ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
                 UBSAN_OPTIONS=halt_on_error=1)
  fi

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
  ar rcsD "${directory}/libg3c4_prompt_t1.a" \
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
  local compiler="${directory}/cxx-wrap"

  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${directory}/cache" \
    ESHKOL_CXX_COMPILER="${compiler}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s "${eshkol_runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      -L "${directory}" --lib g3c4_prompt_t1 \
      "${PROJECT_ROOT}/tests/g3c4/prompt_t1_borrow_test.esk" \
      -o "${directory}/witness" \
      >"${directory}/compile.stdout" 2>"${directory}/compile.stderr"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${directory}/retention-cache" \
    ESHKOL_CXX_COMPILER="${compiler}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s "${eshkol_runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      -L "${directory}" --lib g3c4_prompt_t1 \
      "${PROJECT_ROOT}/tests/g3c4/prompt_t1_borrow_retention.esk" \
      -o "${directory}/retention" \
      >"${directory}/retention-compile.stdout" \
      2>"${directory}/retention-compile.stderr"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${directory}/allocation-cache" \
    ESHKOL_CXX_COMPILER="${compiler}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s "${eshkol_runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      -L "${directory}" --lib g3c4_prompt_t1 \
      "${PROJECT_ROOT}/tests/g3c4/prompt_t1_borrow_allocation_test.esk" \
      -o "${directory}/allocation" \
      >"${directory}/allocation-compile.stdout" \
      2>"${directory}/allocation-compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${directory}/witness" \
    >"${directory}/runtime.stdout" 2>"${directory}/runtime.stderr"
  test ! -s "${directory}/runtime.stderr"
  grep -E '^G3-C4 prompt/T1 borrow PASS: checks=[1-9][0-9]*$' \
    "${directory}/runtime.stdout" >/dev/null
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 300s \
    "${directory}/allocation" \
    >"${directory}/allocation.stdout" \
    2>"${directory}/allocation.stderr"
  test ! -s "${directory}/allocation.stderr"
  grep -E '^G3-C4 prompt/T1 allocation cuts PASS: vector=[1-9][0-9]* cons=[1-9][0-9]* checks=[1-9][0-9]*$' \
    "${directory}/allocation.stdout" >/dev/null
  local horizon
  for horizon in 1024 8192; do
    "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "${directory}/retention" "${horizon}" \
      >"${directory}/retention-${horizon}.stdout" \
      2>"${directory}/retention-${horizon}.stderr"
    test ! -s "${directory}/retention-${horizon}.stderr"
    grep -E "^G3-C4 prompt/T1 retention: horizon=${horizon} input_entries=${horizon} native_tombstones=${horizon} peak_extra_i1=1 live_i1_delta=0 live_i1_bytes_delta=0$" \
      "${directory}/retention-${horizon}.stdout" >/dev/null
  done
}

compile_mode normal
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${tmp}/normal/witness" \
  >"${tmp}/normal/runtime-repeat.stdout" \
  2>"${tmp}/normal/runtime-repeat.stderr"
test ! -s "${tmp}/normal/runtime-repeat.stderr"
cmp "${tmp}/normal/runtime.stdout" "${tmp}/normal/runtime-repeat.stdout"
compile_mode sanitize
cmp "${tmp}/normal/runtime.stdout" "${tmp}/sanitize/runtime.stdout"

for mode in normal sanitize; do
  cp "${tmp}/${mode}/compile.stdout" "${evidence}/${mode}-compile.stdout"
  cp "${tmp}/${mode}/compile.stderr" "${evidence}/${mode}-compile.stderr"
  cp "${tmp}/${mode}/runtime.stdout" "${evidence}/${mode}-runtime.stdout"
  cp "${tmp}/${mode}/runtime.stderr" "${evidence}/${mode}-runtime.stderr"
  cp "${tmp}/${mode}/retention-compile.stdout" \
    "${evidence}/${mode}-retention-compile.stdout"
  cp "${tmp}/${mode}/retention-compile.stderr" \
    "${evidence}/${mode}-retention-compile.stderr"
  cp "${tmp}/${mode}/allocation-compile.stdout" \
    "${evidence}/${mode}-allocation-compile.stdout"
  cp "${tmp}/${mode}/allocation-compile.stderr" \
    "${evidence}/${mode}-allocation-compile.stderr"
  cp "${tmp}/${mode}/allocation.stdout" \
    "${evidence}/${mode}-allocation.stdout"
  cp "${tmp}/${mode}/allocation.stderr" \
    "${evidence}/${mode}-allocation.stderr"
  for horizon in 1024 8192; do
    cp "${tmp}/${mode}/retention-${horizon}.stdout" \
      "${evidence}/${mode}-retention-${horizon}.stdout"
    cp "${tmp}/${mode}/retention-${horizon}.stderr" \
      "${evidence}/${mode}-retention-${horizon}.stderr"
  done
done
cp "${tmp}/normal/runtime-repeat.stdout" "${evidence}/runtime-repeat.stdout"
cp "${PROJECT_ROOT}/native/g3c4_prompt_t1_borrow_source_closure.txt" \
  "${evidence}/source-closure.txt"
while IFS= read -r source; do
  printf '%s\t%s\n' \
    "$(sha256sum "${PROJECT_ROOT}/${source}" | awk '{print $1}')" "${source}"
done <"${PROJECT_ROOT}/native/g3c4_prompt_t1_borrow_source_closure.txt" \
  >"${evidence}/source-sha256.tsv"
sha256sum "${PROJECT_ROOT}/scripts/test-g3c4-prompt-t1-borrow.sh" \
  "${PROJECT_ROOT}/scripts/check-g3c4-prompt-t1-borrow.py" \
  >"${evidence}/runner-sha256.txt"
git -C "${PROJECT_ROOT}" diff --check
cat "${tmp}/normal/runtime.stdout"
printf 'G3-C4 prompt/T1 borrow evidence: %s\n' "${evidence}"
