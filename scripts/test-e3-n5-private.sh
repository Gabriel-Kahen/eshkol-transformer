#!/usr/bin/env bash
# Exact noninstalled N5 edge and stage-2 runtime gate. Run under heavy.lock.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar awk cmp diff env grep nm python3 realpath strings timeout wc; do
  require_command "${command}"
done
[[ -x /usr/bin/time ]] || die "GNU time is required for N5 measurements"
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-n5-private}")"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "N5 output must be fresh"
prefix="${PROJECT_ROOT}/tests/e3_n5/package"
for suffix in sanitized_noninstrumentation_undefined_symbols \
              sanitized_instrumentation_undefined_symbols; do
  [[ -s "${prefix}_${suffix}.txt" ]] ||
    die "N5 sanitizer import inventory is not pinned: ${suffix}"
  LC_ALL=C sort -u "${prefix}_${suffix}.txt" | \
    cmp "${prefix}_${suffix}.txt" - ||
    die "N5 sanitizer inventory is not sorted unique"
done
mkdir -p "${out}"
"${PROJECT_ROOT}/scripts/build-e3-private.sh" "${out}/canonical" \
  >"${out}/canonical.stdout" 2>"${out}/canonical.stderr"
"${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${out}/canonical" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
if grep -E 'et_e3_n5_(edge_)?run_v1' \
    "${out}/canonical/e3_private_package.o.evidence/readelf-symbols.txt" >/dev/null; then
  die "N5 test entry leaked into canonical private object"
fi
for variant in a b sanitized; do
  mode=normal
  [[ "${variant}" != sanitized ]] || mode=sanitize
  "${PROJECT_ROOT}/scripts/build-e3-n5-test.sh" \
    "${out}/package-${variant}" "${mode}" \
    >"${out}/package-${variant}.stdout" \
    2>"${out}/package-${variant}.stderr"
  evidence="${out}/package-${variant}/e3_n5_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    expected=${pair%%:*}
    actual=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${evidence}/${actual}.txt"
  done
  if [[ "${variant}" != sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${evidence}/undefined.txt"
  else
    cmp "${prefix}_sanitized_noninstrumentation_undefined_symbols.txt" \
      "${evidence}/noninstrumentation-undefined.txt"
    cmp "${prefix}_sanitized_instrumentation_undefined_symbols.txt" \
      "${evidence}/sanitizer-undefined.txt"
  fi
  for symbol in et_e3_private_n5_dispatch_cabi_v1 \
                et_e3_private_n5_edge_dispatch_cabi_v1; do
    grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${symbol}$" \
      "${evidence}/readelf-symbols.txt" >/dev/null ||
      die "N5 private dispatch is not local: ${symbol}"
  done
done
generated="${out}/.e3-n5-generated/source/e3_d2_dataset.esk"
for variant in a b; do
  object="${out}/package-${variant}/e3_n5_package.o"
  strings -a "${object}" | grep -Fx "${generated}" >/dev/null ||
    die "N5 generated path is not stable"
done
cmp "${out}/package-a/e3_n5_package.o" \
    "${out}/package-b/e3_n5_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_n5.a" \
    "${out}/package-b/libeshkol_transformer_e3_n5.a"
cmp "${out}/package-a/e3_n5_package.o.evidence/private.d" \
    "${out}/package-b/e3_n5_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_n5_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_n5_package.o.evidence/native-depfiles"

python3 -B -m tests.e3_n5.prepare --output "${out}/corpus" \
  >"${out}/corpus.stdout" 2>"${out}/corpus.stderr"
python3 -B - "${out}/corpus/preflight.json" <<'PY'
import json, pathlib, sys
cases = {row['name']: row for row in json.loads(pathlib.Path(sys.argv[1]).read_text())['cases']}
assert (cases['one-token']['remaining_rows'], cases['one-token']['active_tokens']) == (1, 1)
assert (cases['cap-8192']['remaining_rows'], cases['cap-8192']['active_tokens']) == (8192, 16384)
assert (cases['cap-8193']['remaining_rows'], cases['cap-8193']['active_tokens']) == (8193, 16386)
assert len(cases['cap-8192']['rows']) == 8192
assert len(cases['cap-8193']['rows']) == 8193
expected_payload = {'empty': 224, 'singleton': 496, 'entry-eos': 504,
                    'one-token': 504, 'cap-8192': 133680,
                    'cap-8193': 133696}
for name, expected in expected_payload.items():
    assert sum(file['bytes'] for file in cases[name]['files']) == expected
assert sum(expected_payload.values()) == 269104
PY
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] ||
  die "N5 edge oracle requires absolute pinned Q0_PYTHON"
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
  -m tests.e3_n5.oracle --preflight "${out}/corpus/preflight.json" \
  --output "${out}/one-token-oracle.json" \
  >"${out}/oracle.stdout" 2>"${out}/oracle.stderr"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined \
  -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
runtime_timeout="${E3_N5_RUNTIME_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ &&
   "${runtime_timeout}" =~ ^[1-9][0-9]*$ ]] || die "N5 timeouts must be positive"
for variant in a b sanitized; do
  selected="${cxx}"
  [[ "${variant}" != sanitized ]] || selected="${out}/cxx-sanitized"
  for entry in cap edge; do
    source="${PROJECT_ROOT}/tests/e3_n5/runtime.esk"
    target="${out}/runtime-${variant}"
    [[ "${entry}" != edge ]] || {
      source="${PROJECT_ROOT}/tests/e3_n5/edge_runtime.esk"
      target="${out}/edge-runtime-${variant}"
    }
    env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
      -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
      -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
      -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
      -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${out}/cache-${variant}-${entry}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
      ESHKOL_CXX_COMPILER="${selected}" \
      timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
      "$(eshkol_build_dir)/eshkol-run" --strict-types --no-stdlib -O 2 \
        -L "${out}/package-${variant}" --lib eshkol_transformer_e3_n5 \
        "${source}" -o "${target}" \
        >"${out}/compile-${variant}-${entry}.stdout" \
        2>"${out}/compile-${variant}-${entry}.stderr"
  done
done
cmp "${out}/runtime-a" "${out}/runtime-b"
cmp "${out}/edge-runtime-a" "${out}/edge-runtime-b"
check_time() {
  local path=$1
  [[ "$(wc -l <"${path}")" == 1 ]] || die "N5 runtime measurement is malformed"
  awk -F '[ =]' '$1 == "elapsed_seconds" && $2 ~ /^[0-9]+([.][0-9]+)?$/ &&
                 $3 == "peak_rss_kib" && $4 ~ /^[0-9]+$/ && $4 <= 524288 {
                   accepted = 1
                 }
                 END {exit !accepted}' "${path}" ||
    die "N5 runtime peak RSS exceeds the 524288 KiB ceiling"
}
for variant in a b poison sanitized; do
  executable="${out}/runtime-${variant}"
  [[ "${variant}" != poison ]] || executable="${out}/runtime-a"
  poison=0
  [[ "${variant}" == poison || "${variant}" == sanitized ]] && poison=1
  ESHKOL_ARENA_POISON="${poison}" \
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
      -o "${out}/run-${variant}.time" \
    timeout --foreground --signal=TERM --kill-after=5s "${runtime_timeout}s" \
      "${executable}" "${out}/corpus/one-token" \
      "${out}/corpus/cap-8192" "${out}/corpus/cap-8193" \
      >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  grep -Fx 'E3-N5-CAP-PASS control=8192 stage=2 rows=8193 cursor=unchanged outputs=6' \
    "${out}/run-${variant}.stdout" >/dev/null || die "N5 runtime failed"
  [[ "$(wc -l <"${out}/run-${variant}.stdout")" == 1 &&
     ! -s "${out}/run-${variant}.stderr" ]] || die "N5 runtime emitted diagnostics"
  check_time "${out}/run-${variant}.time"
  cmp "${out}/run-a.stdout" "${out}/run-${variant}.stdout" \
    >/dev/null 2>&1 || [[ "${variant}" == a ]] ||
    die "N5 runtime output differs across modes"
  edge_executable="${out}/edge-runtime-${variant}"
  [[ "${variant}" != poison ]] || edge_executable="${out}/edge-runtime-a"
  ESHKOL_ARENA_POISON="${poison}" \
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
      -o "${out}/edge-${variant}.time" \
    timeout --foreground --signal=TERM --kill-after=5s "${runtime_timeout}s" \
      "${edge_executable}" "${out}/corpus/one-token" \
      "${out}/corpus/empty" "${out}/corpus/singleton" \
      "${out}/corpus/entry-eos" \
      >"${out}/edge-${variant}.stdout" 2>"${out}/edge-${variant}.stderr"
  [[ ! -s "${out}/edge-${variant}.stderr" ]] ||
    die "N5 edge runtime emitted diagnostics"
  check_time "${out}/edge-${variant}.time"
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B \
    -m tests.e3_n5.oracle --oracle "${out}/one-token-oracle.json" \
    --check-run "${out}/edge-${variant}.stdout" \
    >"${out}/edge-${variant}.verify.stdout" \
    2>"${out}/edge-${variant}.verify.stderr"
  cmp "${out}/edge-a.stdout" "${out}/edge-${variant}.stdout" \
    >/dev/null 2>&1 || [[ "${variant}" == a ]] ||
    die "N5 edge runtime output differs across modes"
done
printf 'E3 N5 private edge and stage-2 normal/repeat/poison/sanitizer gate passed\n'
