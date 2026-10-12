#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env find git grep python3 realpath sha256sum sort timeout xargs; do
  require_command "${command}"
done
[[ -x /usr/bin/time ]] || die "GNU time is required for N5 measurements"
[[ $# -eq 1 ]] || die "usage: N5_SOURCE_COMMIT=<exact SHA> $0 FRESH_OUTPUT_DIRECTORY"
[[ "${N5_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || \
  die "N5_SOURCE_COMMIT must pin an exact source commit"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${N5_SOURCE_COMMIT}" ]] || \
  die "N5 source commit differs from exact launch pin"
[[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]] || \
  die "N5 source tree is dirty"
out="$(realpath -m -- "$1")"
[[ ! -e "${out}" ]] || die "N5 gate output directory must be fresh"
mkdir -p "${out}"
printf 'source_commit\t%s\n' "${N5_SOURCE_COMMIT}" >"${out}/pins.tsv"
printf 'source_tree\t%s\n' \
  "$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')" >>"${out}/pins.tsv"
cp "${PROJECT_ROOT}/toolchain/eshkol.lock" "${out}/eshkol.lock"

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.e3_n5.test_generate tests.e3_n5.test_source_contract -v \
  >"${out}/source-tests.stdout" 2>"${out}/source-tests.stderr"
for copy in a b; do
  PYTHONDONTWRITEBYTECODE=1 python3 -m tests.e3_n5.generate \
    --output "${out}/corpus-${copy}" \
    >"${out}/generate-${copy}.stdout" 2>"${out}/generate-${copy}.stderr"
done
diff -r "${out}/corpus-a" "${out}/corpus-b" \
  >"${out}/corpus-diff.stdout"

# This exact noninstalled tuple owns both the existing 17-mode reader and the
# N5-only native entry selectors 29..32. No canonical/installed E3 build is run.
for copy in a b; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-mode-graph-test.sh" \
    "${out}/package-${copy}" normal \
    >"${out}/package-${copy}.stdout" 2>"${out}/package-${copy}.stderr"
done
prefix="${PROJECT_ROOT}/tests/e3_mode_graph/package"
for copy in a b; do
  evidence="${out}/package-${copy}/e3_mode_graph_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings undefined_symbols:undefined \
      source_closure:source-closure native_source_closure:native-source-closure \
      native_objects:native-objects; do
    expected=${pair%%:*}; actual=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${evidence}/${actual}.txt"
  done
  ar t "${out}/package-${copy}/libeshkol_transformer_e3_mode_graph.a" | \
    cmp "${prefix}_archive_members.txt" -
  grep -F 'et_e3_test_retention_probe_v1' "${evidence}/strings.txt" >/dev/null
done
cmp "${out}/package-a/e3_mode_graph_package.o" \
    "${out}/package-b/e3_mode_graph_package.o"
cmp "${out}/package-a/libeshkol_transformer_e3_mode_graph.a" \
    "${out}/package-b/libeshkol_transformer_e3_mode_graph.a"
cmp "${out}/package-a/e3_mode_graph_package.o.evidence/private.d" \
    "${out}/package-b/e3_mode_graph_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_mode_graph_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_mode_graph_package.o.evidence/native-depfiles"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-mode-graph-test.sh" \
  "${out}/package-sanitize" sanitize \
  >"${out}/package-sanitize.stdout" 2>"${out}/package-sanitize.stderr"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cp "${provenance}" "${out}/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "invalid compiler timeout"
for variant in a b sanitize; do
  selected="${cxx}"
  [[ "${variant}" != sanitize ]] || selected="${out}/cxx-sanitized"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${selected}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
      "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_mode_graph \
      "${PROJECT_ROOT}/tests/e3_n5/runtime.esk" -o "${out}/runtime-${variant}" \
      >"${out}/compile-${variant}.stdout" 2>"${out}/compile-${variant}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"

for variant in normal poison sanitize; do
  executable="${out}/runtime-a"
  poison=0
  if [[ "${variant}" == poison ]]; then poison=1; fi
  if [[ "${variant}" == sanitize ]]; then executable="${out}/runtime-sanitize"; poison=1; fi
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON="${poison}" \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
      -o "${out}/run-${variant}.time" \
    timeout --foreground --signal=TERM --kill-after=5s 300s \
      "${executable}" "${out}/corpus-a/one-over" \
      "${out}/corpus-a/retry-small" \
      >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  grep -Fx 'E3-N5-PRIVATE-PASS rows=8193 stage=2 category=unsupported retry=3' \
    "${out}/run-${variant}.stdout" >/dev/null
  [[ ! -s "${out}/run-${variant}.stderr" ]] || \
    die "N5 ${variant} emitted runtime diagnostics"
done
cmp "${out}/run-normal.stdout" "${out}/run-poison.stdout"
cmp "${out}/run-normal.stdout" "${out}/run-sanitize.stdout"
(cd "${out}" && find . -type f ! -name SHA256SUMS -print0 | \
  sort -z | xargs -0 sha256sum >SHA256SUMS && sha256sum -c SHA256SUMS)
printf 'E3 N5 private compiled gate passed: %s\n' "${out}"
