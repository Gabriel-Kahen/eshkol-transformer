#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar awk cat cmp diff env grep python3 realpath sort strings timeout; do require_command "${command}"; done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-mode-graph}")"
[[ ! -e "${out}" ]] || die "E3 mode/graph output directory must be fresh"
mkdir -p "${out}"

python3 -B -m tests.e3_reference.corpus --output "${out}/corpus-a" \
  >"${out}/corpus-a.stdout" 2>"${out}/corpus-a.stderr"
python3 -B -m tests.e3_reference.corpus --output "${out}/corpus-b" \
  >"${out}/corpus-b.stdout" 2>"${out}/corpus-b.stderr"
diff -r "${out}/corpus-a" "${out}/corpus-b"

"${PROJECT_ROOT}/scripts/build-e3-private.sh" "${out}/canonical" \
  >"${out}/canonical.stdout" 2>"${out}/canonical.stderr"
"${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${out}/canonical" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
if grep -Eq 'e3-mode-graph|et_e3_test_graph_event_v1|p1-e3-test-mode-mask' \
    "${out}/canonical/e3_private_package.o.evidence/strings.txt"; then
  die "mode/graph observer leaked into canonical private object"
fi

prefix="${PROJECT_ROOT}/tests/e3_mode_graph/package"
for variant in a b missed-child enter-write-only graph-create sanitized; do
  mode="${variant}"
  case "${variant}" in a|b) mode=normal ;; sanitized) mode=sanitize ;; esac
  "${PROJECT_ROOT}/scripts/build-e3-mode-graph-test.sh" \
    "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" \
    2>"${out}/package-${variant}.stderr"
  artifact="${out}/package-${variant}/e3_mode_graph_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    expected=${pair%%:*}; actual=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${artifact}/${actual}.txt"
  done
  if [[ "${variant}" != sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${artifact}/undefined.txt"
  else
    cmp "${prefix}_sanitized_noninstrumentation_undefined_symbols.txt" \
      "${artifact}/noninstrumentation-undefined.txt"
    cmp "${prefix}_sanitized_instrumentation_undefined_symbols.txt" \
      "${artifact}/sanitizer-undefined.txt"
    [[ "$(awk 'END {print NR}' "${artifact}/undefined.txt")" == 223 ]] || \
      die "mode/graph sanitized total import count differs from sealed tuple"
    cat "${artifact}/noninstrumentation-undefined.txt" \
        "${artifact}/sanitizer-undefined.txt" | LC_ALL=C sort -u | \
      cmp "${artifact}/undefined.txt" -
  fi
  grep -F 'et_e3_test_graph_event_v1' "${artifact}/strings.txt" >/dev/null || \
    die "mode/graph owner observer absent from ${variant}"
done
generated_source="${out}/.e3-mode-graph-generated/source/e3_d2_dataset.esk"
for variant in a b; do
  object="${out}/package-${variant}/e3_mode_graph_package.o"
  strings -a "${object}" | grep -Fx "${generated_source}" >/dev/null ||
    die "mode/graph ${variant} object omitted its stable runtime source filename"
  if strings -a "${object}" | grep -F '/eshkol-transformer-e1b.' >/dev/null; then
    die "mode/graph ${variant} object retained a random generated source path"
  fi
done
cmp "${out}/package-a/e3_mode_graph_package.o" \
    "${out}/package-b/e3_mode_graph_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_mode_graph.a" \
    "${out}/package-b/libeshkol_transformer_e3_mode_graph.a"
cmp "${out}/package-a/e3_mode_graph_package.o.evidence/private.d" \
    "${out}/package-b/e3_mode_graph_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_mode_graph_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_mode_graph_package.o.evidence/native-depfiles"

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
  die "E3 mode/graph compiler timeout must be positive"
for variant in a b missed-child enter-write-only graph-create sanitized; do
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
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_mode_graph \
      "${PROJECT_ROOT}/tests/e3_mode_graph/runtime.esk" \
      -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" \
      2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"

for gradient in absent present; do
  for variant in a b poisoned sanitized missed-child graph-create; do
    executable="${variant}"
    poison=0
    if [[ "${variant}" == poisoned ]]; then executable=a; poison=1; fi
    if [[ "${variant}" == sanitized ]]; then
      ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON=1 \
        timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/runtime-${executable}" "${out}/corpus-a/packed-single" \
        "${gradient}" >"${out}/run-${variant}-${gradient}.stdout" \
        2>"${out}/run-${variant}-${gradient}.stderr"
    else
      ESHKOL_ARENA_POISON="${poison}" \
        timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/runtime-${executable}" "${out}/corpus-a/packed-single" \
        "${gradient}" >"${out}/run-${variant}-${gradient}.stdout" \
        2>"${out}/run-${variant}-${gradient}.stderr"
    fi
    [[ ! -s "${out}/run-${variant}-${gradient}.stderr" ]] || \
      die "mode/graph ${variant}/${gradient} emitted runtime diagnostics"
    expected=normal
    case "${variant}" in missed-child|graph-create) expected="${variant}" ;; esac
    python3 -B -m tests.e3_mode_graph.verify_observation \
      --stdout "${out}/run-${variant}-${gradient}.stdout" \
      --expect "${expected}" \
      >"${out}/verify-${variant}-${gradient}.stdout" \
      2>"${out}/verify-${variant}-${gradient}.stderr"
  done
  cmp "${out}/run-a-${gradient}.stdout" "${out}/run-b-${gradient}.stdout"
  cmp "${out}/run-a-${gradient}.stdout" "${out}/run-poisoned-${gradient}.stdout"
  cmp "${out}/run-a-${gradient}.stdout" "${out}/run-sanitized-${gradient}.stdout"
  for mutant in missed-child graph-create; do
    if python3 -B -m tests.e3_mode_graph.verify_observation \
        --stdout "${out}/run-${mutant}-${gradient}.stdout" --expect normal \
        >"${out}/kill-${mutant}-${gradient}.stdout" \
        2>"${out}/kill-${mutant}-${gradient}.stderr"; then
      die "compiled ${mutant} mutant survived the normal in-call observer"
    fi
  done
  grep -F 'in-call mode-mask differs' \
    "${out}/kill-missed-child-${gradient}.stderr" >/dev/null || \
    die "missed-child mutant died outside actual in-call mode assertion"
  grep -F 'in-call graph-event differs' \
    "${out}/kill-graph-create-${gradient}.stderr" >/dev/null || \
    die "graph-create mutant died outside real graph event assertion"
done

if ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/runtime-enter-write-only" "${out}/corpus-a/packed-single" \
      absent >"${out}/run-enter-write-only.stdout" \
      2>"${out}/run-enter-write-only.stderr"; then
  die "enter-write-only P1 fail-stop control survived"
else
  status=$?
  [[ "${status}" == 134 ]] || \
    die "enter-write-only control failed outside the expected abort path: ${status}"
  if grep -F 'E3-MODE-GRAPH-RUNTIME-PASS' \
      "${out}/run-enter-write-only.stdout" >/dev/null; then
    die "enter-write-only control published success"
  fi
fi
printf 'E3 in-call mode/graph witness and compiled mutants passed\n'
