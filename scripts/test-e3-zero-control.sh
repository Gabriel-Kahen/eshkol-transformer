#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep python3 realpath strings timeout; do
  require_command "${command}"
done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-zero-control}")"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "output must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] || \
  die "E3 zero-control oracle requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  "${q0_python}" - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || \
  die "E3 zero-control oracle versions are not pinned"
for cache in a b; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
    -m tests.e3_private.zero_control_reference \
    --output "${out}/oracle-${cache}" \
    >"${out}/oracle-${cache}.stdout" 2>"${out}/oracle-${cache}.stderr"
done
diff -r "${out}/oracle-a" "${out}/oracle-b"

"${PROJECT_ROOT}/scripts/build-e3-private.sh" "${out}/canonical" \
  >"${out}/canonical.stdout" 2>"${out}/canonical.stderr"
"${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${out}/canonical" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
if grep -F 'E3ROLE' \
  "${out}/canonical/e3_private_package.o.evidence/strings.txt" >/dev/null; then
  die "zero-control observer leaked into canonical private object"
fi

prefix="${PROJECT_ROOT}/tests/e3_private/zero_control_package"
for variant in a b; do
  "${PROJECT_ROOT}/scripts/build-e3-zero-control-test.sh" \
    "${out}/package-${variant}" normal \
    >"${out}/package-${variant}.stdout" \
    2>"${out}/package-${variant}.stderr"
  artifact="${out}/package-${variant}/e3_zero_control_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects \
      undefined_symbols:undefined; do
    expected=${pair%%:*}; actual=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${artifact}/${actual}.txt"
  done
  grep -F 'E3ROLE' "${artifact}/strings.txt" >/dev/null || \
    die "zero-control observer is absent from ${variant}"
  object="${out}/package-${variant}/e3_zero_control_package.o"
  strings -a "${object}" | \
    grep -Fx "${out}/.e3-zero-control-generated/source/e3_d2_dataset.esk" \
    >/dev/null || die "zero-control object omitted stable source filename"
  if strings -a "${object}" | grep -F '/eshkol-transformer-e1b.' >/dev/null; then
    die "zero-control object retained a random generated source filename"
  fi
done
cmp "${out}/package-a/e3_zero_control_package.o" \
    "${out}/package-b/e3_zero_control_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_zero_control.a" \
    "${out}/package-b/libeshkol_transformer_e3_zero_control.a"
cmp "${out}/package-a/e3_zero_control_package.o.evidence/private.d" \
    "${out}/package-b/e3_zero_control_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_zero_control_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_zero_control_package.o.evidence/native-depfiles"

"${PROJECT_ROOT}/scripts/build-e3-zero-control-test.sh" \
  "${out}/package-sanitize" sanitize \
  >"${out}/package-sanitize.stdout" \
  2>"${out}/package-sanitize.stderr"
san_artifact="${out}/package-sanitize/e3_zero_control_package.o.evidence"
for pair in defined_symbols:global-defined public_exports:package-exports \
    public_strings:public-strings source_closure:source-closure \
    native_source_closure:native-source-closure native_objects:native-objects; do
  expected=${pair%%:*}; actual=${pair#*:}
  cmp "${prefix}_${expected}.txt" "${san_artifact}/${actual}.txt"
done
cmp "${prefix}_sanitized_noninstrumentation_undefined_symbols.txt" \
    "${san_artifact}/noninstrumentation-undefined.txt"
cmp "${prefix}_sanitized_instrumentation_undefined_symbols.txt" \
    "${san_artifact}/sanitizer-undefined.txt"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
printf '#!/usr/bin/env bash\nexec %q -fsanitize=address,undefined -fno-omit-frame-pointer "$@"\n' \
  "${cxx}" >"${out}/sanitizer-cxx"
chmod 700 "${out}/sanitizer-cxx"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E3 zero-control compiler timeout must be positive"
for variant in a b; do
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
      "${runner}" --strict-types --no-stdlib -O 2 \
        -L "${out}/package-${variant}" --lib eshkol_transformer_e3_zero_control \
        "${PROJECT_ROOT}/tests/e3_private/zero_control_runtime.esk" \
        -o "${out}/runtime-${variant}" \
        >"${out}/compile-${variant}.stdout" \
        2>"${out}/compile-${variant}.stderr"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-negative-${variant}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
      "${runner}" --strict-types --no-stdlib -O 2 \
        -L "${out}/package-${variant}" --lib eshkol_transformer_e3_zero_control \
        "${PROJECT_ROOT}/tests/e3_private/zero_control_negative_runtime.esk" \
        -o "${out}/negative-runtime-${variant}" \
        >"${out}/negative-compile-${variant}.stdout" \
        2>"${out}/negative-compile-${variant}.stderr"
done
for variant in a b; do
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-sanitize-${variant}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${out}/sanitizer-cxx" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
      "${runner}" --strict-types --no-stdlib -O 1 \
        -L "${out}/package-sanitize" --lib eshkol_transformer_e3_zero_control \
        "${PROJECT_ROOT}/tests/e3_private/zero_control_runtime.esk" \
        -o "${out}/runtime-sanitize-${variant}" \
        >"${out}/compile-sanitize-${variant}.stdout" \
        2>"${out}/compile-sanitize-${variant}.stderr"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-sanitize-negative-${variant}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${out}/sanitizer-cxx" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
      "${runner}" --strict-types --no-stdlib -O 1 \
        -L "${out}/package-sanitize" --lib eshkol_transformer_e3_zero_control \
        "${PROJECT_ROOT}/tests/e3_private/zero_control_negative_runtime.esk" \
        -o "${out}/negative-runtime-sanitize-${variant}" \
        >"${out}/negative-compile-sanitize-${variant}.stdout" \
        2>"${out}/negative-compile-sanitize-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"
cmp "${out}/negative-runtime-a" "${out}/negative-runtime-b"
cmp "${out}/runtime-sanitize-a" "${out}/runtime-sanitize-b"
cmp "${out}/negative-runtime-sanitize-a" "${out}/negative-runtime-sanitize-b"
for case_name in all-correct all-wrong mixed; do
  for gradient in absent present; do
    for variant in a b sanitize-a sanitize-b; do
      runtime="${out}/runtime-${variant}"
      ESHKOL_ARENA_POISON=0 \
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        LSAN_OPTIONS=exitcode=23 \
        timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${runtime}" \
        "${out}/oracle-a/${case_name}" "${case_name}" "${gradient}" \
        >"${out}/run-${case_name}-${gradient}-${variant}.stdout" \
        2>"${out}/run-${case_name}-${gradient}-${variant}.stderr"
      [[ ! -s "${out}/run-${case_name}-${gradient}-${variant}.stderr" ]] || \
        die "zero-control runtime emitted diagnostics"
      python3 -m tests.e3_private.parse_zero_control_observation \
        --stdout "${out}/run-${case_name}-${gradient}-${variant}.stdout" \
        --observed "${out}/observed-${case_name}-${gradient}-${variant}.json" \
        --roles "${out}/roles-${case_name}-${gradient}-${variant}.tsv"
      ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
        -m tests.e3_native_parity.verify_observation \
        --reference "${out}/oracle-a/reference.json" --case "${case_name}" \
        --observed "${out}/observed-${case_name}-${gradient}-${variant}.json" \
        --roles "${out}/roles-${case_name}-${gradient}-${variant}.tsv" \
        --transcript "${out}/transcript-${case_name}-${gradient}-${variant}.json" \
        >"${out}/verify-${case_name}-${gradient}-${variant}.stdout" \
        2>"${out}/verify-${case_name}-${gradient}-${variant}.stderr"
    done
    cmp "${out}/run-${case_name}-${gradient}-a.stdout" \
        "${out}/run-${case_name}-${gradient}-b.stdout"
    cmp "${out}/transcript-${case_name}-${gradient}-a.json" \
        "${out}/transcript-${case_name}-${gradient}-b.json"
    cmp "${out}/run-${case_name}-${gradient}-a.stdout" \
        "${out}/run-${case_name}-${gradient}-sanitize-a.stdout"
    cmp "${out}/run-${case_name}-${gradient}-sanitize-a.stdout" \
        "${out}/run-${case_name}-${gradient}-sanitize-b.stdout"
    cmp "${out}/transcript-${case_name}-${gradient}-a.json" \
        "${out}/transcript-${case_name}-${gradient}-sanitize-a.json"
    cmp "${out}/transcript-${case_name}-${gradient}-sanitize-a.json" \
        "${out}/transcript-${case_name}-${gradient}-sanitize-b.json"
  done
done
for case_name in missing-path foreign-handle foreign-tree stale-handle \
    duplicate-tied-head topology-path-10 topology-shape-10 topology-tie-10 \
    topology-throw-path-10 \
    source-alias wrong-shape wrong-rank wrong-f32-geometry wrong-kind \
    forged-source released-source decode-short decode-extent decode-role \
    decode-request-length decode-payload decode-shape decode-rank-limit \
    decode-dtype decode-device decode-layout decode-tail decode-operation \
    commit-reuse active-borrow active-pin \
    alloc-i2-decode alloc-f32-decode alloc-i2-prepare alloc-f32-prepare; do
  for gradient in absent present; do
    for variant in a b sanitize-a sanitize-b; do
      ESHKOL_ARENA_POISON=0 \
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        LSAN_OPTIONS=exitcode=23 \
        timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/negative-runtime-${variant}" \
        "${out}/oracle-a/all-correct" "${case_name}" "${gradient}" \
        >"${out}/negative-${case_name}-${gradient}-${variant}.stdout" \
        2>"${out}/negative-${case_name}-${gradient}-${variant}.stderr"
      [[ ! -s "${out}/negative-${case_name}-${gradient}-${variant}.stderr" ]] || \
        die "zero-control ${case_name}/${gradient} emitted diagnostics"
      grep -Fx 'E3-ZERO-NEGATIVE-RUNTIME-PASS' \
        "${out}/negative-${case_name}-${gradient}-${variant}.stdout" >/dev/null || \
        die "zero-control ${case_name}/${gradient} did not complete"
    done
    cmp "${out}/negative-${case_name}-${gradient}-a.stdout" \
        "${out}/negative-${case_name}-${gradient}-b.stdout"
    cmp "${out}/negative-${case_name}-${gradient}-a.stdout" \
        "${out}/negative-${case_name}-${gradient}-sanitize-a.stdout"
    cmp "${out}/negative-${case_name}-${gradient}-sanitize-a.stdout" \
        "${out}/negative-${case_name}-${gradient}-sanitize-b.stdout"
  done
done
printf 'E3 real zero-model and preprepare controls passed\n'
