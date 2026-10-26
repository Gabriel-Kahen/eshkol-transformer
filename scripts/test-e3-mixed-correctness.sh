#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep python3 realpath strings timeout; do require_command "${command}"; done
[[ $# -eq 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "${out}" ]] || die "N3 mixed output directory must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] || die "N3 mixed requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || die "N3 mixed oracle versions are not pinned"
for variant in a b; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_reference.mixed_correctness --output "${out}/corpus-${variant}" \
    >"${out}/corpus-${variant}.stdout" 2>"${out}/corpus-${variant}.stderr"
done
diff -r "${out}/corpus-a" "${out}/corpus-b"
cmp "${out}/corpus-a-reference.json" "${out}/corpus-b-reference.json"

prefix="${PROJECT_ROOT}/tests/e3_private/numerical_package"
for variant in a b sanitized; do
  mode=normal
  [[ "${variant}" == sanitized ]] && mode=sanitize
  "${PROJECT_ROOT}/scripts/build-e3-numerical-test.sh" "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" 2>"${out}/package-${variant}.stderr"
  artifact="${out}/package-${variant}/e3_numerical_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    cmp "${prefix}_${pair%%:*}.txt" "${artifact}/${pair#*:}.txt"
  done
  if [[ "${variant}" != sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${artifact}/undefined.txt"
  fi
  grep -F 'E3ROW' "${artifact}/strings.txt" >/dev/null || die "${variant}: numerical observer absent"
done
cmp "${out}/package-a/e3_numerical_package.o" "${out}/package-b/e3_numerical_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_numerical.a" "${out}/package-b/libeshkol_transformer_e3_numerical.a"
cmp "${out}/package-a/e3_numerical_package.o.evidence/private.d" \
    "${out}/package-b/e3_numerical_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_numerical_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_numerical_package.o.evidence/native-depfiles"
generated_source="${out}/.e3-numerical-generated/source/e3_d2_dataset.esk"
for variant in a b; do
  object="${out}/package-${variant}/e3_numerical_package.o"
  strings -a "${object}" | grep -Fx "${generated_source}" >/dev/null || \
    die "${variant}: stable generated source path absent"
  if strings -a "${object}" | grep -F '/eshkol-transformer-e1b.' >/dev/null; then
    die "${variant}: random generated source path retained"
  fi
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "N3 mixed compiler timeout must be positive"
for variant in a b sanitized; do
  cxx_selected="${cxx}"
  [[ "${variant}" == sanitized ]] && cxx_selected="${out}/cxx-sanitized"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx_selected}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_numerical \
      "${PROJECT_ROOT}/tests/e3_private/mixed_correctness_runtime.esk" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" 2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"

for variant in a b sanitized poison; do
  runtime="${variant}" poison=0
  [[ "${variant}" == poison ]] && runtime=a && poison=1
  for case_name in matched-correct mixed; do
    if [[ "${variant}" == sanitized ]]; then
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
        ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/runtime-${runtime}" "${case_name}" "${out}/corpus-a/${case_name}" \
        >"${out}/run-${variant}-${case_name}.stdout" 2>"${out}/run-${variant}-${case_name}.stderr"
    else
      ESHKOL_ARENA_POISON="${poison}" timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/runtime-${runtime}" "${case_name}" "${out}/corpus-a/${case_name}" \
        >"${out}/run-${variant}-${case_name}.stdout" 2>"${out}/run-${variant}-${case_name}.stderr"
    fi
    stdout="${out}/run-${variant}-${case_name}.stdout"
    [[ ! -s "${out}/run-${variant}-${case_name}.stderr" ]] || die "${variant}-${case_name}: runtime diagnostics"
    python3 -B -m tests.e3_private.parse_mixed_correctness_observation \
      --stdout "${stdout}" --corpus "${out}/corpus-a" --case "${case_name}" \
      --observed "${out}/observed-${variant}-${case_name}.json" \
      --roles "${out}/roles-${variant}-${case_name}.tsv" \
      --result "${out}/result-${variant}-${case_name}.json"
    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
      -m tests.e3_native_parity.verify_observation \
      --reference "${out}/corpus-a-reference.json" --case "${case_name}" \
      --observed "${out}/observed-${variant}-${case_name}.json" \
      --roles "${out}/roles-${variant}-${case_name}.tsv" \
      --transcript "${out}/transcript-${variant}-${case_name}.json" \
      >"${out}/verify-${variant}-${case_name}.stdout" \
      2>"${out}/verify-${variant}-${case_name}.stderr"
  done
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_private.verify_mixed_correctness \
    --reference "${out}/corpus-a-reference.json" --corpus "${out}/corpus-a" \
    --stdout-matched-correct "${out}/run-${variant}-matched-correct.stdout" \
    --stdout-mixed "${out}/run-${variant}-mixed.stdout" \
    --observed-matched-correct "${out}/observed-${variant}-matched-correct.json" \
    --observed-mixed "${out}/observed-${variant}-mixed.json" \
    --roles-matched-correct "${out}/roles-${variant}-matched-correct.tsv" \
    --roles-mixed "${out}/roles-${variant}-mixed.tsv" \
    --result-matched-correct "${out}/result-${variant}-matched-correct.json" \
    --result-mixed "${out}/result-${variant}-mixed.json" \
    --report "${out}/comparison-${variant}.json"
done
for variant in b poison; do
  cmp "${out}/comparison-a.json" "${out}/comparison-${variant}.json" || \
    die "${variant}: paired report differs"
  for case_name in matched-correct mixed; do
    cmp "${out}/run-a-${case_name}.stdout" "${out}/run-${variant}-${case_name}.stdout"
    cmp "${out}/transcript-a-${case_name}.json" "${out}/transcript-${variant}-${case_name}.json"
  done
done
printf 'N3 mixed-correctness genuine one-row D2 transcripts passed\n'
