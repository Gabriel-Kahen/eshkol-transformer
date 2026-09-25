#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp env grep python3 sha256sum timeout; do
  require_command "${command}"
done
python3 "${PROJECT_ROOT}/scripts/check-g3c4-t1-eval-admission.py"

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-t1-eval-admission-test}"
mkdir -p "${evidence}"

if [[ "${G3C4_SKIP_CALL_ENTRY_REGRESSION:-0}" == 1 ]]; then
  printf 'G3-C4 call-entry regression: SKIPPED\n' \
    >"${evidence}/call-entry-regression.stdout"
else
  "${PROJECT_ROOT}/scripts/test-g3c4-call-entry.sh" \
    "${evidence}/call-entry-regression" \
    >"${evidence}/call-entry-regression.stdout" 2>&1
fi

common_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer"
  -I "$(eshkol_source_dir)/inc"
)

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-t1-eval.XXXXXX")"
cleanup() {
  local status=$?
  if (( status == 0 )); then
    rm -rf -- "${tmp}"
  else
    printf 'G3-C4 T1/eval failure evidence retained at %s\n' "${tmp}" >&2
    find "${tmp}" -type f \
      \( -name 'compile.stderr' -o -name 'runtime.stderr' -o -name 'runtime.stdout' \) \
      -print -exec cat {} \; >&2 || true
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
    -c "${PROJECT_ROOT}/tests/g3c4/call_entry_native.c" \
    -o "${directory}/objects/g3c4_call_entry_native.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -c "${PROJECT_ROOT}/tests/g3c4/call_entry_a2_stats.c" \
    -o "${directory}/objects/g3c4_call_entry_a2_stats.o"
  "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
    -DET_P1_TRUSTED_BUILD=1 -c "${PROJECT_ROOT}/native/p1_identity.c" \
    -o "${directory}/objects/p1_identity.o"
  local source
  for source in data_io checkpoint_io kernel_abi t1_i64_shell \
                n3k_primitives_provider n2_primitives_provider \
                a2_attention_provider; do
    "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
      -DET_A2_KV_CACHE_TESTING -c "${PROJECT_ROOT}/native/${source}.c" \
      -o "${directory}/objects/${source}.o"
  done
  for source in m3_i64_integration m3_model; do
    "${cc}" "${common_flags[@]}" "${mode_flags[@]}" \
      -c "${PROJECT_ROOT}/src/eshkol_transformer/${source}.c" \
      -o "${directory}/objects/${source}.o"
  done
  ar rcsD "${directory}/libg3c4_t1_eval.a" "${directory}"/objects/*.o

  local compiler="${cxx}"
  if [[ "${mode}" == sanitize ]]; then
    cat >"${directory}/sanitize-cxx" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
    chmod 0500 "${directory}/sanitize-cxx"
    compiler="${directory}/sanitize-cxx"
  fi

  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${directory}/cache" \
    ESHKOL_CXX_COMPILER="${compiler}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    timeout --foreground --signal=TERM --kill-after=5s 600s "${runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      -L "${directory}" --lib g3c4_t1_eval \
      "${PROJECT_ROOT}/tests/g3c4/t1_eval_admission_test.esk" \
      -o "${directory}/witness" \
      >"${directory}/compile.stdout" 2>"${directory}/compile.stderr"
  "${runtime[@]}" ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${directory}/witness" \
    >"${directory}/runtime.stdout" 2>"${directory}/runtime.stderr"
  test ! -s "${directory}/runtime.stderr"
  grep -E '^G3-C4 T1/eval admission PASS: checks=[1-9][0-9]*$' \
    "${directory}/runtime.stdout" >/dev/null
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
done
cp "${tmp}/normal/runtime-repeat.stdout" "${evidence}/runtime-repeat.stdout"
cp "${PROJECT_ROOT}/native/g3c4_t1_eval_admission_source_closure.txt" \
  "${evidence}/source-closure.txt"
while IFS= read -r source; do
  printf '%s\t%s\n' \
    "$(sha256sum "${PROJECT_ROOT}/${source}" | awk '{print $1}')" "${source}"
done <"${PROJECT_ROOT}/native/g3c4_t1_eval_admission_source_closure.txt" \
  >"${evidence}/source-sha256.tsv"
sha256sum "${PROJECT_ROOT}/scripts/test-g3c4-t1-eval-admission.sh" \
  "${PROJECT_ROOT}/scripts/check-g3c4-t1-eval-admission.py" \
  >"${evidence}/runner-sha256.txt"
git -C "${PROJECT_ROOT}" diff --check
cat "${tmp}/normal/runtime.stdout"
printf 'G3-C4 T1/eval admission evidence: %s\n' "${evidence}"
