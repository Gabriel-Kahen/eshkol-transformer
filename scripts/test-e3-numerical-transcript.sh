#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep python3 realpath strings timeout; do require_command "${command}"; done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-numerical-transcript}")"
[[ ! -e "${out}" ]] || die "E3 numerical output directory must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] || \
  die "E3 numerical proof requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || \
  die "E3 numerical oracle versions are not pinned: ${versions//$'\n'/ }"
for cache in a b; do
  "${q0_python}" -m tests.e3_reference.corpus \
    --output "${out}/corpus-${cache}" \
    >"${out}/corpus-${cache}.stdout" 2>"${out}/corpus-${cache}.stderr"
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
    -m tests.e3_reference.reference \
    --corpus-reference "${out}/corpus-${cache}/corpus-reference.json" \
    --output "${out}/reference-${cache}.json" \
    >"${out}/reference-${cache}.stdout" 2>"${out}/reference-${cache}.stderr"
done
diff -r "${out}/corpus-a" "${out}/corpus-b"
cmp "${out}/reference-a.json" "${out}/reference-b.json"

# The ordinary private package is built independently and its object must not
# contain the frame emitter. The numerical tuple keeps its own exact manifests.
"${PROJECT_ROOT}/scripts/build-e3-private.sh" "${out}/canonical" \
  >"${out}/canonical.stdout" 2>"${out}/canonical.stderr"
"${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${out}/canonical" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
if grep -F 'E3ROW' \
    "${out}/canonical/e3_private_package.o.evidence/strings.txt" >/dev/null; then
  die "numerical observer leaked into canonical private object"
fi

prefix="${PROJECT_ROOT}/tests/e3_private/numerical_package"
for variant in a b wrong-target mean-of-means second-normalization sanitized; do
  mode="${variant}"
  case "${variant}" in a|b) mode=normal ;; sanitized) mode=sanitize ;; esac
  "${PROJECT_ROOT}/scripts/build-e3-numerical-test.sh" \
    "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" \
    2>"${out}/package-${variant}.stderr"
  artifact="${out}/package-${variant}/e3_numerical_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    expected=${pair%%:*}; actual=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${artifact}/${actual}.txt"
  done
  if [[ "${variant}" != sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${artifact}/undefined.txt"
  fi
  grep -F 'E3ROW' "${artifact}/strings.txt" >/dev/null || \
    die "numerical frame observer is absent from ${variant}"
done
generated_source="${out}/.e3-numerical-generated/source/e3_d2_dataset.esk"
for variant in a b; do
  object="${out}/package-${variant}/e3_numerical_package.o"
  strings -a "${object}" | grep -Fx "${generated_source}" >/dev/null ||
    die "numerical ${variant} object omitted its stable runtime source filename"
  if strings -a "${object}" | grep -F '/eshkol-transformer-e1b.' >/dev/null; then
    die "numerical ${variant} object retained a random generated source path"
  fi
done
cmp "${out}/package-a/e3_numerical_package.o" \
    "${out}/package-b/e3_numerical_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_numerical.a" \
    "${out}/package-b/libeshkol_transformer_e3_numerical.a"
cmp "${out}/package-a/e3_numerical_package.o.evidence/private.d" \
    "${out}/package-b/e3_numerical_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_numerical_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_numerical_package.o.evidence/native-depfiles"

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
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E3 numerical compiler timeout must be positive"
for variant in a b wrong-target mean-of-means second-normalization sanitized; do
  cxx_selected="${cxx}"
  if [[ "${variant}" == sanitized ]]; then
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
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_numerical \
      "${PROJECT_ROOT}/tests/e3_private/numerical_runtime.esk" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" \
      2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"
for variant in a b wrong-target mean-of-means second-normalization sanitized; do
  if [[ "${variant}" == sanitized ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 \
      ESHKOL_ARENA_POISON=1 \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/runtime-${variant}" "${out}/corpus-a/packed-single" \
      >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  else
    ESHKOL_ARENA_POISON=0 \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/runtime-${variant}" "${out}/corpus-a/packed-single" \
      >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  fi
  grep -Fx 'E3-NUMERICAL-RUNTIME-PASS' "${out}/run-${variant}.stdout" \
    >/dev/null || die "compiled numerical ${variant} did not complete"
  [[ ! -s "${out}/run-${variant}.stderr" ]] || \
    die "compiled numerical ${variant} emitted runtime diagnostics"
  python3 -m tests.e3_private.parse_numerical_observation \
    --stdout "${out}/run-${variant}.stdout" \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv"
done
ESHKOL_ARENA_POISON=1 \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${out}/runtime-a" "${out}/corpus-a/packed-single" \
  >"${out}/run-poison.stdout" 2>"${out}/run-poison.stderr"
cmp "${out}/run-a.stdout" "${out}/run-poison.stdout"
[[ ! -s "${out}/run-poison.stderr" ]] || \
  die "poisoned numerical runtime emitted diagnostics"
python3 -m tests.e3_private.parse_numerical_observation \
  --stdout "${out}/run-poison.stdout" \
  --observed "${out}/observed-poison.json" \
  --roles "${out}/roles-poison.tsv"
cmp "${out}/run-a.stdout" "${out}/run-b.stdout"
for variant in a b poison sanitized; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
    -m tests.e3_native_parity.verify_observation \
    --reference "${out}/reference-a.json" --case packed-single \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv" \
    --transcript "${out}/transcript-${variant}.json" \
    >"${out}/verify-${variant}.stdout" 2>"${out}/verify-${variant}.stderr"
done
cmp "${out}/transcript-a.json" "${out}/transcript-b.json"
for variant in wrong-target mean-of-means second-normalization; do
  if ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
    -m tests.e3_native_parity.verify_observation \
    --reference "${out}/reference-a.json" --case packed-single \
    --observed "${out}/observed-${variant}.json" \
    --roles "${out}/roles-${variant}.tsv" \
    --transcript "${out}/transcript-${variant}.json" \
    >"${out}/verify-${variant}.stdout" 2>"${out}/verify-${variant}.stderr"; then
    die "compiled numerical ${variant} mutant survived"
  fi
  [[ ! -e "${out}/transcript-${variant}.json" ]] || \
    die "failed numerical mutant published a transcript"
done
grep -F 'ce_f32' "${out}/verify-wrong-target.stderr" >/dev/null || \
  die "wrong-target mutant failed outside the CE value assertion"
grep -F 'state_f32' "${out}/verify-mean-of-means.stderr" >/dev/null || \
  die "mean-of-means mutant failed outside the accumulator value assertion"
grep -F 'final.metrics_f32' "${out}/verify-second-normalization.stderr" >/dev/null || \
  die "second-normalization mutant failed outside the final loss value assertion"
printf 'E3 compiled three-batch numerical transcript and three mutants passed\n'
