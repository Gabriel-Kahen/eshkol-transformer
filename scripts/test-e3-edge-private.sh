#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
retention_builder="${PROJECT_ROOT}/scripts/build-e3-retention-test.sh"
[[ -f "${retention_builder}" && -x "${retention_builder}" ]] || \
  die "E3 retention test builder is missing or not executable: ${retention_builder}"
verify_toolchain
for command in ar cmp diff env grep nm python3 realpath timeout; do
  require_command "${command}"
done
[[ -x /usr/bin/time ]] || die "GNU time is required for edge resource evidence"
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-edge-private}")"
[[ ! -e "${out}" ]] || die "edge output directory must be fresh: ${out}"
mkdir -p "${out}"

# Freeze authentic D1 bytes and independent D2 cursor/row expectations twice.
for cache in a b; do
  for edge_case in empty singleton entry-eos one-token-tail; do
    python3 -B -m tests.e3_horizon.edge_generate \
      --output "${out}/corpus-${cache}-${edge_case}" --case "${edge_case}" \
      >"${out}/generate-${cache}-${edge_case}.stdout" \
      2>"${out}/generate-${cache}-${edge_case}.stderr"
  done
done
for edge_case in empty singleton entry-eos one-token-tail; do
  diff -r "${out}/corpus-a-${edge_case}" "${out}/corpus-b-${edge_case}"
  cmp "${out}/generate-a-${edge_case}.stdout" \
      "${out}/generate-b-${edge_case}.stdout"
done
cat "${out}"/generate-a-{empty,singleton,entry-eos,one-token-tail}.stdout \
  >"${out}/oracle.tsv"

# The edge phase exists only in this noninstalled test aggregate. Its bridge and
# retention observers must not enter the canonical private object.
"${PROJECT_ROOT}/scripts/build-e3-private.sh" "${out}/canonical" \
  >"${out}/canonical.stdout" 2>"${out}/canonical.stderr"
"${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${out}/canonical" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
if nm -g --defined-only "${out}/canonical/e3_private_package.o" | \
    grep -E ' et_e3_retention_run_v1$'; then
  die "edge test bridge leaked into canonical private object"
fi

prefix="${PROJECT_ROOT}/tests/e3_private/retention_package"
for cache in a b sanitized; do
  mode=normal
  [[ "${cache}" != sanitized ]] || mode=sanitize
  "${retention_builder}" \
    "${out}/private-${cache}" "${mode}" \
    >"${out}/package-${cache}.stdout" 2>"${out}/package-${cache}.stderr"
  artifact="${out}/private-${cache}/e3_retention_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    cmp "${prefix}_${pair%%:*}.txt" "${artifact}/${pair#*:}.txt"
  done
  if [[ "${cache}" != sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${artifact}/undefined.txt"
  fi
done
cmp "${out}/private-a/e3_retention_package.o" \
    "${out}/private-b/e3_retention_package.o"
cmp "${out}/private-a/libeshkol_transformer_e3_retention.a" \
    "${out}/private-b/libeshkol_transformer_e3_retention.a"
cmp "${out}/private-a/e3_retention_package.o.evidence/private.d" \
    "${out}/private-b/e3_retention_package.o.evidence/private.d"
diff -r "${out}/private-a/e3_retention_package.o.evidence/native-depfiles" \
        "${out}/private-b/e3_retention_package.o.evidence/native-depfiles"
source "${PROJECT_ROOT}/scripts/e3-retention-sanitizer-import-policy.sh"
e3_retention_verify_sanitizer_imports \
  "${prefix}_undefined_symbols.txt" \
  "${out}/private-sanitized/e3_retention_package.o.evidence/noninstrumentation-undefined.txt" \
  "${out}/private-sanitized/e3_retention_package.o.evidence/sanitizer-undefined.txt"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -fstack-protector-all \
  -fPIC -fvisibility=hidden -fno-common \
  -I "$(eshkol_source_dir)/lib/core" -I "$(eshkol_source_dir)/inc" \
  -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
  -o "${out}/retention_arena.o"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -fstack-protector-all \
  -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -fPIC -fvisibility=hidden -fno-common \
  -I "$(eshkol_source_dir)/lib/core" -I "$(eshkol_source_dir)/inc" \
  -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
  -o "${out}/retention_arena_sanitized.o"
for cache in a b sanitized; do
  mkdir "${out}/link-${cache}"
  supplement="${out}/retention_arena.o"
  [[ "${cache}" != sanitized ]] || supplement="${out}/retention_arena_sanitized.o"
  ar p "${out}/private-${cache}/libeshkol_transformer_e3_retention.a" \
    e3_retention_package.o >"${out}/link-${cache}/package.o"
  cmp "${out}/private-${cache}/e3_retention_package.o" \
      "${out}/link-${cache}/package.o"
  ar rcsD "${out}/link-${cache}/libeshkol_transformer_e3_retention_test.a" \
    "${out}/link-${cache}/package.o" "${supplement}"
done
cmp "${out}/link-a/libeshkol_transformer_e3_retention_test.a" \
    "${out}/link-b/libeshkol_transformer_e3_retention_test.a"

cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined \
  -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
runtime_timeout="${E3_EDGE_RUNTIME_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ && \
   "${runtime_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "edge timeouts must be positive integers"
for cache in a b sanitized; do
  cxx_selected="${cxx}"
  [[ "${cache}" != sanitized ]] || cxx_selected="${out}/cxx-sanitized"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${cache}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx_selected}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/link-${cache}" --lib eshkol_transformer_e3_retention_test \
      "${PROJECT_ROOT}/tests/e3_private/edge_runtime.esk" \
      -o "${out}/runtime-${cache}" \
      >"${out}/compile-${cache}.stdout" 2>"${out}/compile-${cache}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"

# Both injected exceptions must fail through the boxed bridge with distinct,
# flushed pre/post evaluator provenance. They never count as edge successes.
for control in pre post; do
  if timeout --foreground --signal=TERM --kill-after=5s \
      "${runtime_timeout}s" "${out}/runtime-a" \
      "${out}/corpus-a-empty" "${out}/corpus-a-singleton" \
      "${out}/corpus-a-entry-eos" "${out}/corpus-a-one-token-tail" \
      "--edge-${control}-fault" \
      >"${out}/control-${control}.stdout" \
      2>"${out}/control-${control}.stderr"; then
    die "edge ${control} exception control unexpectedly passed"
  else
    status=$?
  fi
  [[ "${status}" -eq 1 ]] || die "edge ${control} control exited ${status}, expected 1"
  python3 -B -m tests.e3_private.edge_phase_check \
    --stderr "${out}/control-${control}.stderr" --mode "${control}"
  [[ ! -s "${out}/control-${control}.stdout" ]] || \
    die "edge ${control} control emitted success output"
done

for profile in normal-a normal-b poisoned-a sanitized; do
  executable="${out}/runtime-a"
  poison=0
  [[ "${profile}" != normal-b ]] || executable="${out}/runtime-b"
  if [[ "${profile}" == poisoned-a ]]; then poison=1; fi
  if [[ "${profile}" == sanitized ]]; then
    executable="${out}/runtime-sanitized"
    poison=1
  fi
  ESHKOL_ARENA_POISON="${poison}" \
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
      -o "${out}/${profile}.time" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${runtime_timeout}s" "${executable}" \
      "${out}/corpus-a-empty" "${out}/corpus-a-singleton" \
      "${out}/corpus-a-entry-eos" "${out}/corpus-a-one-token-tail" \
      >"${out}/${profile}.stdout" 2>"${out}/${profile}.stderr"
  grep -Fx 'E3-EDGE-PRIVATE-PASS cases=4 zero-weight=3 one-token-tail=1' \
    "${out}/${profile}.stdout" >/dev/null || die "edge ${profile} did not pass"
  python3 -B -m tests.e3_horizon.edge_accuracy \
    --oracle "${out}/oracle.tsv" --stdout "${out}/${profile}.stdout"
  python3 -B -m tests.e3_private.edge_phase_check \
    --stderr "${out}/${profile}.stderr" --mode normal
  grep -Eq '^elapsed_seconds=[0-9.]+ peak_rss_kib=[1-9][0-9]*$' \
    "${out}/${profile}.time" || die "edge ${profile} resource record is absent"
done
for profile in normal-b poisoned-a sanitized; do
  cmp "${out}/normal-a.stdout" "${out}/${profile}.stdout"
done
printf 'E3 private edge traversal and atomicity gate passed: %s\n' "${out}"
