#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp cp diff env git grep python3 realpath sha256sum timeout; do
  require_command "${command}"
done
[[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" != 1 ]] || \
  die "N5 acceptance gate requires the supported host; compatibility override is forbidden"
[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_FRESH_OUTPUT_DIRECTORY (with N5_* pins)"
for name in N5_SOURCE_COMMIT N5_SOURCE_TREE N5_LOCK_SHA256 \
    N5_PROVENANCE_SHA256 N5_RUNNER_SHA256 N5_CC_SHA256 N5_CXX_SHA256 \
    N5_CORPUS_INVENTORY_SHA256; do
  [[ "${!name:-}" =~ ^[0-9a-f]{40}$ || \
     "${!name:-}" =~ ^[0-9a-f]{64}$ ]] || die "invalid ${name} pin"
done
[[ "${N5_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ && \
   "${N5_SOURCE_TREE}" =~ ^[0-9a-f]{40}$ ]] || die "source pins require full Git IDs"
for name in N5_LOCK_SHA256 N5_PROVENANCE_SHA256 N5_RUNNER_SHA256 \
    N5_CC_SHA256 N5_CXX_SHA256 N5_CORPUS_INVENTORY_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "${name} requires SHA-256"
done
[[ "$1" == /* && "$1" == "$(realpath -m -s -- "$1")" && \
   "$1" == "$(realpath -m -- "$1")" ]] || \
  die "N5 output path must be absolute, canonical and free of symlink ancestors"
out=$1
[[ "${out}" != "${PROJECT_ROOT}" && "${out}" != "${PROJECT_ROOT}/"* ]] || \
  die "N5 output must be outside the source checkout"
[[ ! -e "${out}" && ! -L "${out}" ]] || \
  die "N5 gate output directory must be fresh"
mkdir -p "${out}"
stage=pin-preflight
pin_hash() { sha256sum "$1" | awk '{print $1}'; }
check_pins() {
  local provenance runner cc cxx key expected actual
  provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
  runner="$(eshkol_build_dir)/eshkol-run"
  [[ -r "${provenance}" && -x "${runner}" ]] || return 1
  cc="$(tsv_value "${provenance}" cc_path)" || return 1
  cxx="$(tsv_value "${provenance}" cxx_path)" || return 1
  [[ "${cc}" == /* && "${cxx}" == /* && -x "${cc}" && -x "${cxx}" ]] || return 1
  [[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${N5_SOURCE_COMMIT}" && \
     "$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')" == "${N5_SOURCE_TREE}" && \
     -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]] || return 1
  for key in lock provenance runner cc cxx; do
    case "${key}" in
      lock) expected=${N5_LOCK_SHA256}; actual="$(pin_hash "${PROJECT_ROOT}/toolchain/eshkol.lock")" ;;
      provenance) expected=${N5_PROVENANCE_SHA256}; actual="$(pin_hash "${provenance}")" ;;
      runner) expected=${N5_RUNNER_SHA256}; actual="$(pin_hash "${runner}")" ;;
      cc) expected=${N5_CC_SHA256}; actual="$(pin_hash "${cc}")" ;;
      cxx) expected=${N5_CXX_SHA256}; actual="$(pin_hash "${cxx}")" ;;
    esac
    printf '%s\t%s\t%s\n' "${key}" "${expected}" "${actual}"
    [[ "${actual}" == "${expected}" ]] || return 1
  done
  printf 'source_commit\t%s\nsource_tree\t%s\n' \
    "${N5_SOURCE_COMMIT}" "${N5_SOURCE_TREE}"
}
fixture_check() {
  local path=$1 actual
  actual="$(PYTHONPATH="${PROJECT_ROOT}" PYTHONDONTWRITEBYTECODE=1 \
    python3 -m tests.e3_n5.evidence fixture-digest "${path}")" || return 1
  printf '%s\t%s\t%s\n' "${path}" "${N5_CORPUS_INVENTORY_SHA256}" "${actual}"
  [[ "${actual}" == "${N5_CORPUS_INVENTORY_SHA256}" ]]
}
finalize() {
  local status=$? seal_status=0 verify_status=0 pin_status fixture_status copy
  trap - EXIT INT TERM
  set +e
  check_pins >"${out}/postflight-pins.tsv" 2>"${out}/postflight-pins.stderr"
  pin_status=$?
  printf '%d\n' "${pin_status}" >"${out}/postflight-pins.status"
  if [[ ${pin_status} -ne 0 && ${status} -eq 0 ]]; then
    stage=postflight-pins
    status=1
  fi
  for copy in a b; do
    if [[ -d "${out}/corpus-${copy}" ]]; then
      fixture_check "${out}/corpus-${copy}" \
        >"${out}/postflight-fixture-${copy}.tsv" \
        2>"${out}/postflight-fixture-${copy}.stderr"
      fixture_status=$?
      printf '%d\n' "${fixture_status}" >"${out}/postflight-fixture-${copy}.status"
      if [[ ${fixture_status} -ne 0 && ${status} -eq 0 ]]; then
        stage="postflight-fixture-${copy}"
        status=1
      fi
    else
      printf 'missing corpus-%s at postflight\n' "${copy}" \
        >"${out}/postflight-fixture-${copy}.stderr"
      printf '1\n' >"${out}/postflight-fixture-${copy}.status"
      if [[ ${status} -eq 0 ]]; then
        stage="postflight-fixture-${copy}"
        status=1
      fi
    fi
  done
  PYTHONPATH="${PROJECT_ROOT}" PYTHONDONTWRITEBYTECODE=1 \
    python3 -m tests.e3_n5.evidence seal "${out}" "${status}" "${stage}"
  seal_status=$?
  if [[ -e "${out}/SEAL-COLLISION.json" || -L "${out}/SEAL-COLLISION.json" ]]; then
    printf 'N5 seal collision after gate status %d; reserved evidence was preserved\n' \
      "${status}" >&2
    status=73
  elif [[ ! -f "${out}/SHA256SUMS" || -L "${out}/SHA256SUMS" ]]; then
    printf 'N5 seal was not completed after gate status %d (seal status %d)\n' \
      "${status}" "${seal_status}" >&2
    status=74
  else
    [[ ${status} -ne 0 ]] || status=${seal_status}
    (cd "${out}" && timeout --foreground --signal=TERM --kill-after=5s 300s \
      bash -c 'sha256sum -c --strict --status SHA256SUMS && \
        PYTHONPATH="$1" PYTHONDONTWRITEBYTECODE=1 \
        python3 -m tests.e3_n5.evidence verify-inventory .' _ "${PROJECT_ROOT}")
    verify_status=$?
    if [[ ${verify_status} -ne 0 ]]; then
      printf 'N5 post-seal checksum/inventory verification failed (status %d)\n' \
        "${verify_status}" >&2
      status=74
    fi
  fi
  if [[ ${status} -eq 0 ]]; then
    printf 'E3 N5 private compiled gate passed: %s\n' "${out}"
  else
    printf 'E3 N5 gate failed at %s; evidence retained at %s (status %d)\n' \
      "${stage}" "${out}" "${status}" >&2
  fi
  exit "${status}"
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
check_pins >"${out}/preflight-pins.tsv" 2>"${out}/preflight-pins.stderr" || \
  die "N5 reviewed source/toolchain/compiler pins differ"
cp "${PROJECT_ROOT}/toolchain/eshkol.lock" "${out}/eshkol.lock"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" \
  "${out}/eshkol-transformer-provenance.tsv"
printf 'corpus_inventory_sha256\t%s\n' "${N5_CORPUS_INVENTORY_SHA256}" \
  >"${out}/fixture-pin.tsv"
stage=verify-toolchain
verify_toolchain
[[ -x /usr/bin/time ]] || die "GNU time is required for N5 measurements"

stage=source-tests
PYTHONPATH="${PROJECT_ROOT}" PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.e3_n5.test_generate tests.e3_n5.test_source_contract \
  tests.e3_n5.test_evidence tests.e3_mode_graph.test_mode_graph_tuple -v \
  >"${out}/source-tests.stdout" 2>"${out}/source-tests.stderr"
for copy in a b; do
  stage="generate-${copy}"
  PYTHONPATH="${PROJECT_ROOT}" PYTHONDONTWRITEBYTECODE=1 \
    python3 -m tests.e3_n5.generate \
    --output "${out}/corpus-${copy}" \
    >"${out}/generate-${copy}.stdout" 2>"${out}/generate-${copy}.stderr"
done
stage=fixture-preflight
diff -r "${out}/corpus-a" "${out}/corpus-b" \
  >"${out}/corpus-diff.stdout"
for copy in a b; do
  fixture_check "${out}/corpus-${copy}" >"${out}/fixture-${copy}.tsv" \
    2>"${out}/fixture-${copy}.stderr" || die "N5 ${copy} fixture differs from reviewed pin"
done

# This exact noninstalled tuple owns both the existing 17-mode reader and the
# N5-only native entry selectors 29..32. No canonical/installed E3 build is run.
for copy in a b; do
  stage="package-${copy}"
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-mode-graph-test.sh" \
    "${out}/package-${copy}" normal \
    >"${out}/package-${copy}.stdout" 2>"${out}/package-${copy}.stderr"
done
stage=package-manifest
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

stage=package-sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-mode-graph-test.sh" \
  "${out}/package-sanitize" sanitize \
  >"${out}/package-sanitize.stdout" 2>"${out}/package-sanitize.stderr"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
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
  stage="compile-${variant}"
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
  stage="run-${variant}"
  executable="${out}/runtime-a"
  poison=0
  if [[ "${variant}" == poison ]]; then poison=1; fi
  if [[ "${variant}" == sanitize ]]; then executable="${out}/runtime-sanitize"; poison=1; fi
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON="${poison}" \
    /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
      -o "${out}/run-${variant}.time" \
    timeout --foreground --signal=TERM --kill-after=5s 3600s \
      "${executable}" "${out}/corpus-a/one-over" \
      "${out}/corpus-a/retry-small" "${out}/corpus-a/exact-limit" \
      >"${out}/run-${variant}.stdout" 2>"${out}/run-${variant}.stderr"
  printf '%s\n' \
    'E3-N5-PRIVATE-PASS rows=8193 stage=2 category=unsupported retry=3 exact=8192 active=16383 final=255/0/1/0' | \
    cmp - "${out}/run-${variant}.stdout"
  [[ ! -s "${out}/run-${variant}.stderr" ]] || \
    die "N5 ${variant} emitted runtime diagnostics"
done
stage=runtime-comparison
cmp "${out}/run-normal.stdout" "${out}/run-poison.stdout"
cmp "${out}/run-normal.stdout" "${out}/run-sanitize.stdout"
stage=complete
