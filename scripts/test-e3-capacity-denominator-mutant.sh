#!/usr/bin/env bash
# Noninstalled compiled-private M2 capacity-denominator gate.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in cmp diff env grep python3 realpath strings timeout; do require_command "${command}"; done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-capacity-denominator-mutant}")"
[[ ! -e "${out}" ]] || die "capacity-denominator output directory must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] || die "absolute pinned Q0_PYTHON required"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || die "capacity-denominator Q0 versions differ"
E3_SOURCE_CC="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)" \
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
  -m unittest tests.e3_reference.test_sum_order_witness \
    tests.e3_private.test_capacity_denominator_mutant \
  >"${out}/source.stdout" 2>"${out}/source.stderr"
for copy in a b; do
  python3 -B -m tests.e3_private.sum_order_corpus \
    --output "${out}/corpus-${copy}" \
    >"${out}/corpus-${copy}.stdout" 2>"${out}/corpus-${copy}.stderr"
done
diff -r "${out}/corpus-a" "${out}/corpus-b"

"${PROJECT_ROOT}/scripts/build-e3-private.sh" "${out}/canonical" \
  >"${out}/canonical.stdout" 2>"${out}/canonical.stderr"
"${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${out}/canonical" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
if grep -F 'E3ROW' "${out}/canonical/e3_private_package.o.evidence/strings.txt" >/dev/null; then
  die "numerical observer leaked into canonical private object"
fi

prefix="${PROJECT_ROOT}/tests/e3_private/numerical_package"
variants=(normal-a normal-b mutant-a mutant-b normal-sanitize mutant-sanitize)
for variant in "${variants[@]}"; do
  mode=normal
  case "${variant}" in
    mutant-a|mutant-b) mode=capacity-denominator ;;
    normal-sanitize) mode=sanitize ;;
    mutant-sanitize) mode=capacity-denominator-sanitize ;;
  esac
  "${PROJECT_ROOT}/scripts/build-e3-numerical-test.sh" \
    "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" 2>"${out}/package-${variant}.stderr"
  evidence="${out}/package-${variant}/e3_numerical_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    cmp "${prefix}_${pair%%:*}.txt" "${evidence}/${pair#*:}.txt"
  done
  if [[ "${variant}" != *-sanitize ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${evidence}/undefined.txt"
  fi
  grep -F 'E3ROW' "${evidence}/strings.txt" >/dev/null || \
    die "numerical observer missing from ${variant}"
done
for name in normal mutant; do
  cmp "${out}/package-${name}-a/e3_numerical_package.o" \
      "${out}/package-${name}-b/e3_numerical_package.o"
  cmp "${out}/package-${name}-a/libeshkol_transformer_e3_numerical.a" \
      "${out}/package-${name}-b/libeshkol_transformer_e3_numerical.a"
  cmp "${out}/package-${name}-a/e3_numerical_package.o.evidence/private.d" \
      "${out}/package-${name}-b/e3_numerical_package.o.evidence/private.d"
  diff -r "${out}/package-${name}-a/e3_numerical_package.o.evidence/native-depfiles" \
          "${out}/package-${name}-b/e3_numerical_package.o.evidence/native-depfiles"
done

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined \
  -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "invalid compiler timeout"
for variant in "${variants[@]}"; do
  cxx_selected="${cxx}"
  [[ "${variant}" != *-sanitize ]] || cxx_selected="${out}/cxx-sanitized"
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
      "${PROJECT_ROOT}/tests/e3_private/numerical_runtime.esk" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" 2>"${out}/compile-${variant}.stderr"
done
for name in normal mutant; do
  cmp "${out}/runtime-${name}-a" "${out}/runtime-${name}-b"
done

run_case() {
  local variant=$1 poison=$2 key=$3
  local parse_args=()
  local case_name=sum-order
  if [[ "${variant}" == mutant* ]]; then
    parse_args=(--capacity-mutant)
    case_name=sum-order-capacity
  fi
  if [[ "${variant}" == *-sanitize ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/runtime-${variant}" "${case_name}" "${out}/corpus-a" \
      >"${out}/run-${key}.stdout" 2>"${out}/run-${key}.stderr"
  else
    ESHKOL_ARENA_POISON="${poison}" \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/runtime-${variant}" "${case_name}" "${out}/corpus-a" \
      >"${out}/run-${key}.stdout" 2>"${out}/run-${key}.stderr"
  fi
  grep -Fx 'E3-NUMERICAL-RUNTIME-PASS' "${out}/run-${key}.stdout" >/dev/null ||
    die "${key} did not complete all three genuine batches"
  [[ ! -s "${out}/run-${key}.stderr" ]] || die "${key} emitted runtime diagnostics"
  python3 -B -m tests.e3_private.parse_numerical_observation \
    --stdout "${out}/run-${key}.stdout" --batch-count 3 \
    --observed "${out}/observed-${key}.json" \
    --roles "${out}/roles-${key}.tsv" "${parse_args[@]}"
}
for variant in "${variants[@]}"; do
  run_case "${variant}" 0 "${variant}"
done
for name in normal mutant; do
  run_case "${name}-a" 1 "${name}-poison"
  for suffix in b poison; do
    cmp "${out}/run-${name}-a.stdout" "${out}/run-${name}-${suffix}.stdout"
  done
done

verify_normal() {
  local key=$1
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_private.verify_capacity_denominator_mutant \
    --observed "${out}/observed-${key}.json" \
    --roles "${out}/roles-${key}.tsv" \
    --transcript "${out}/transcript-${key}.json" \
    >"${out}/verify-${key}.stdout" 2>"${out}/verify-${key}.stderr"
  grep -Fx 'E3-CAPACITY-DENOMINATOR-NORMAL-PASS' "${out}/verify-${key}.stdout" >/dev/null
}
for key in normal-a normal-b normal-poison normal-sanitize; do
  verify_normal "${key}"
done
cmp "${out}/transcript-normal-a.json" "${out}/transcript-normal-b.json"
cmp "${out}/transcript-normal-a.json" "${out}/transcript-normal-poison.json"

for pair in mutant-a:normal-a mutant-b:normal-b \
            mutant-poison:normal-poison mutant-sanitize:normal-sanitize; do
  mutant=${pair%%:*}; baseline=${pair#*:}
  if ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_private.verify_capacity_denominator_mutant \
    --observed "${out}/observed-${mutant}.json" \
    --roles "${out}/roles-${mutant}.tsv" \
    --baseline-observed "${out}/observed-${baseline}.json" \
    --baseline-roles "${out}/roles-${baseline}.tsv" \
    --transcript "${out}/transcript-${mutant}.json" \
    >"${out}/verify-${mutant}.stdout" 2>"${out}/verify-${mutant}.stderr"; then
    die "${mutant} survived the genuine capacity-denominator assertion"
  fi
  grep -F 'batches[2].reduction_f32: exact rational reduction mismatch' \
    "${out}/verify-${mutant}.stderr" >/dev/null ||
    die "${mutant} failed before the required third-row reduction comparator"
  [[ ! -e "${out}/transcript-${mutant}.json" ]] ||
    die "${mutant} published a rejected transcript"
done
printf 'E3 genuine capacity-denominator compiled-private mutant gate passed\n'
