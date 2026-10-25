#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep python3 realpath strings timeout; do
  require_command "${command}"
done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-padding-hit-mutant}")"
[[ ! -e "${out}" && ! -L "${out}" ]] ||
  die "E3 padding-hit output directory must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] ||
  die "E3 padding-hit proof requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] ||
  die "E3 padding-hit oracle versions are not pinned: ${versions//$'\n'/ }"
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  tests.e3_private.test_padding_hit_source \
  tests.e3_private.test_uniform_tuple \
  >"${out}/source.stdout" 2>"${out}/source.stderr"
for copy in a b; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_reference.padding_hit \
    --output "${out}/fixture-${copy}" \
    --reference "${out}/reference-${copy}.json" \
    >"${out}/reference-${copy}.stdout" \
    2>"${out}/reference-${copy}.stderr"
done
diff -r "${out}/fixture-a" "${out}/fixture-b"
cmp "${out}/reference-a.json" "${out}/reference-b.json"

# The canonical private package must retain its own exact output boundary.
"${PROJECT_ROOT}/scripts/build-e3-private.sh" "${out}/canonical" \
  >"${out}/canonical.stdout" 2>"${out}/canonical.stderr"
"${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${out}/canonical" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
if grep -F 'E3ROW' \
    "${out}/canonical/e3_private_package.o.evidence/strings.txt" >/dev/null; then
  die "uniform frame observer leaked into the canonical private object"
fi
prefix="${PROJECT_ROOT}/tests/e3_private/uniform_package"
for variant in a b mutant-a mutant-b sanitized mutant-sanitized; do
  mode="${variant}"
  case "${variant}" in a|b) mode=normal ;; mutant-a|mutant-b) mode=padding-hit ;; sanitized) mode=sanitize ;; mutant-sanitized) mode=padding-hit-sanitize ;; esac
  "${PROJECT_ROOT}/scripts/build-e3-uniform-test.sh" \
    "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" \
    2>"${out}/package-${variant}.stderr"
  evidence="${out}/package-${variant}/e3_uniform_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    expected=${pair%%:*}; actual=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${evidence}/${actual}.txt"
  done
  if [[ "${variant}" != *sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${evidence}/undefined.txt"
  fi
  grep -F 'E3ROW' "${evidence}/strings.txt" >/dev/null ||
    die "uniform frame observer is absent from ${variant}"
  if grep -E '^[[:space:]]*[0-9a-fA-F]+[[:space:]]+GLOBAL[[:space:]].*[[:space:]]et_i2_test_' \
      "${evidence}/readelf-symbols.txt" >/dev/null; then
    die "I2 fault hooks escaped uniform package localization"
  fi
done
generated="${out}/.e3-uniform-generated/source/e3_d2_dataset.esk"
for variant in a b; do
  object="${out}/package-${variant}/e3_uniform_package.o"
  strings -a "${object}" | grep -Fx "${generated}" >/dev/null ||
    die "uniform ${variant} object omitted its stable generated source"
  if strings -a "${object}" | grep -F '/eshkol-transformer-e1b.' >/dev/null; then
    die "uniform ${variant} object retained a random generated source path"
  fi
done
cmp "${out}/package-a/e3_uniform_package.o" \
    "${out}/package-b/e3_uniform_package.o"
cmp "${out}/package-mutant-a/e3_uniform_package.o" \
    "${out}/package-mutant-b/e3_uniform_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_uniform.a" \
    "${out}/package-b/libeshkol_transformer_e3_uniform.a"
cmp "${out}/package-mutant-a/libeshkol_transformer_e3_uniform.a" \
    "${out}/package-mutant-b/libeshkol_transformer_e3_uniform.a"
cmp "${out}/package-a/e3_uniform_package.o.evidence/private.d" \
    "${out}/package-b/e3_uniform_package.o.evidence/private.d"
cmp "${out}/package-mutant-a/e3_uniform_package.o.evidence/private.d" \
    "${out}/package-mutant-b/e3_uniform_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_uniform_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_uniform_package.o.evidence/native-depfiles"
diff -r "${out}/package-mutant-a/e3_uniform_package.o.evidence/native-depfiles" \
        "${out}/package-mutant-b/e3_uniform_package.o.evidence/native-depfiles"

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
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] ||
  die "E3 padding-hit compiler timeout must be positive"
for variant in a b mutant-a mutant-b sanitized mutant-sanitized; do
  cxx_selected="${cxx}"
  if [[ "${variant}" == *sanitized ]]; then
    cxx_selected="${out}/cxx-sanitized"
  fi
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx_selected}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_uniform \
      "${PROJECT_ROOT}/tests/e3_private/uniform_runtime.esk" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" \
      2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"
cmp "${out}/runtime-mutant-a" "${out}/runtime-mutant-b"

run_case() {
  local variant=$1 runtime=$1 case_name=padding-hit-normal poison=0
  case "${variant}" in
    mutant-a|mutant-b|mutant-sanitized) case_name=padding-hit-mutant ;;
    poison) runtime=a; poison=1 ;;
    mutant-poison) runtime=mutant-a; case_name=padding-hit-mutant; poison=1 ;;
  esac
  if [[ "${variant}" == *sanitized ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/runtime-${runtime}" "${case_name}" \
      "${out}/fixture-a/padding-hit" \
      "${out}/fixture-a/zero-state.etcp" \
      "${out}/fixture-a/bad-checksum.etcp" \
      "${out}/fixture-a/bad-provider.etcp" \
      "${out}/fixture-a/bad-shape.etcp" \
      "${out}/fixture-a/bad-tie.etcp" \
      >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  else
    ESHKOL_ARENA_POISON="${poison}" \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/runtime-${runtime}" "${case_name}" \
      "${out}/fixture-a/padding-hit" \
      "${out}/fixture-a/zero-state.etcp" \
      "${out}/fixture-a/bad-checksum.etcp" \
      "${out}/fixture-a/bad-provider.etcp" \
      "${out}/fixture-a/bad-shape.etcp" \
      "${out}/fixture-a/bad-tie.etcp" \
      >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  fi
  [[ ! -s "${out}/run-${variant}.stderr" ]] ||
    die "padding-hit ${variant} emitted runtime diagnostics"
  python3 -B -m tests.e3_private.parse_uniform_observation \
    --stdout "${out}/run-${variant}.stdout" --batch-count 3 \
    --reference "${out}/reference-a.json" \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv"
}
for variant in a b mutant-a mutant-b poison mutant-poison sanitized mutant-sanitized; do
  run_case "${variant}"
done
for pair in a:b a:poison mutant-a:mutant-b mutant-a:mutant-poison; do
  cmp "${out}/run-${pair%%:*}.stdout" "${out}/run-${pair#*:}.stdout"
done
for variant in a b poison sanitized; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_private.verify_padding_hit \
    --reference "${out}/reference-a.json" \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv" \
    --transcript "${out}/transcript-${variant}.json" \
    >"${out}/verify-${variant}.stdout" 2>"${out}/verify-${variant}.stderr"
  grep -Fx 'E3-PADDING-HIT-NORMAL-PASS' "${out}/verify-${variant}.stdout" >/dev/null
done
cmp "${out}/transcript-a.json" "${out}/transcript-b.json"
cmp "${out}/transcript-a.json" "${out}/transcript-poison.json"
for pair in mutant-a:a mutant-b:b mutant-poison:poison mutant-sanitized:sanitized; do
  mutant=${pair%%:*}; baseline=${pair#*:}
  if ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_private.verify_padding_hit \
    --reference "${out}/reference-a.json" \
    --observed "${out}/observed-${mutant}.json" \
    --roles "${out}/roles-${mutant}.tsv" \
    --baseline-observed "${out}/observed-${baseline}.json" \
    --baseline-roles "${out}/roles-${baseline}.tsv" \
    --transcript "${out}/transcript-${mutant}.json" \
    >"${out}/verify-${mutant}.stdout" 2>"${out}/verify-${mutant}.stderr"; then
    die "padding-hit ${mutant} survived the independent correctness oracle"
  fi
  grep -F 'batches[2].correct_f32' "${out}/verify-${mutant}.stderr" >/dev/null ||
    die "padding-hit ${mutant} failed before the final correctness word"
  [[ ! -e "${out}/transcript-${mutant}.json" ]] ||
    die "padding-hit ${mutant} published a rejected transcript"
done
printf 'E3 genuine padding-hit compiled-private mutant gate passed\n'
