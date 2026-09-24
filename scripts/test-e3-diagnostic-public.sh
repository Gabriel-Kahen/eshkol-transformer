#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env grep nm python3 timeout; do require_command "${command}"; done
output="${1:-$(project_build_dir)/e3-diagnostic-public-gate}"
mkdir -p "${output}"
artifact="${output}/installed"
prefix="${PROJECT_ROOT}/native/e3_diagnostic_public_package"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-diagnostic-public.sh" "${artifact}"
for pair in 'defined_symbols global-defined' 'public_exports package-exports' \
    'public_strings public-strings' 'undefined_symbols undefined' \
    'source_closure source-closure' 'native_source_closure native-source-closure' \
    'native_objects native-objects'; do
  read -r expected observed <<<"${pair}"
  cmp "${prefix}_${expected}.txt" \
    "${artifact}/e3_diagnostic_public_package.o.evidence/${observed}.txt"
done
(cd "${artifact}/facades" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
  >"${output}/facades.txt"
cmp "${prefix}_facades.txt" "${output}/facades.txt"
ar t "${artifact}/libeshkol_transformer_e3_diagnostic.a" \
  >"${output}/archive-members.txt"
cmp "${prefix}_archive_members.txt" "${output}/archive-members.txt"

corpus="${output}/corpus"
python3 -m tests.e3_reference.corpus --output "${corpus}"
reference="${E3_REFERENCE_BUNDLE:-}"
[[ "${reference}" == /* && -f "${reference}" ]] || \
  die "E3_REFERENCE_BUNDLE must name the accepted absolute reference bundle"
python3 - "${corpus}/corpus-reference.json" "${reference}" <<'PY'
from pathlib import Path
import sys
from tests.e3_reference.transcript import decode
corpus = decode(Path(sys.argv[1]).read_bytes())
reference = decode(Path(sys.argv[2]).read_bytes())
if ({case["name"]: case for case in corpus["cases"]} !=
    {case["name"]: case["corpus"] for case in reference["cases"]}):
    raise SystemExit("accepted E3 reference does not bind the generated corpus")
PY

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
compile() {
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${output}/cache" ESHKOL_LIB_DIR="${artifact}/facades" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -I "${artifact}/facades" "$@"
}
for source in compile_api public_runtime quota_runtime; do
  compile --compile-only --emit-depfile "${output}/${source}.d" \
    "${PROJECT_ROOT}/tests/e3_diagnostic_public/${source}.esk" \
    -o "${output}/${source}.o"
  python3 "${PROJECT_ROOT}/tests/e3_diagnostic_public/check_public_closure.py" \
    "${artifact}" "${PROJECT_ROOT}/tests/e3_diagnostic_public/${source}.esk" \
    "${output}/${source}.d"
done
if nm -u --format=posix "${output}/compile_api.o" | \
    grep -E 'et_e1b_private_|et_e3_test_|e3-diagnostic-report-stats'; then
  die "installed E3 facade retained private or test authority"
fi
nm -u --format=posix "${output}/compile_api.o" | \
  awk '$1 ~ /^et_e1b_public_e3_/ {print $1}' | LC_ALL=C sort -u \
  >"${output}/api-e3-undefined.txt"
grep '^et_e1b_public_e3_' "${prefix}_public_exports.txt" \
  >"${output}/expected-api-e3-undefined.txt"
cmp "${output}/expected-api-e3-undefined.txt" "${output}/api-e3-undefined.txt"
compile -O 2 -L "${artifact}" --lib eshkol_transformer_e3_diagnostic \
  "${PROJECT_ROOT}/tests/e3_diagnostic_public/public_runtime.esk" \
  -o "${output}/public-runtime"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 600s \
  "${output}/public-runtime" "${corpus}/packed-single" \
  >"${output}/public-runtime.stdout" \
  2>"${output}/public-runtime.stderr"
grep -Fx 'E3-DIAGNOSTIC-PUBLIC-PASS' "${output}/public-runtime.stdout" >/dev/null

compile -O 2 -L "${artifact}" --lib eshkol_transformer_e3_diagnostic \
  "${PROJECT_ROOT}/tests/e3_diagnostic_public/quota_runtime.esk" \
  -o "${output}/quota-runtime"
quota_timeout="${E3_PUBLIC_QUOTA_TIMEOUT_SECONDS:-900}"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  "${quota_timeout}s" "${output}/quota-runtime" "${corpus}/packed-single" \
  >"${output}/quota-runtime.stdout" \
  2>"${output}/quota-runtime.stderr"
grep -Fx \
  'E3-DIAGNOSTIC-PUBLIC-QUOTA-PASS reservations=8192 successes=8191 failures=1' \
  "${output}/quota-runtime.stdout" >/dev/null

for symbol in \
    et_e1b_private_e3_diagnostic_evaluate_fixed_cabi_v1 \
    et_e1b_private_e3_diagnostic_evaluation_f32_bits_cabi_v1 \
    et_e3_diagnostic_destination_create_v1 \
    et_e3_diagnostic_test_run_v1; do
  printf '(extern ptr hidden :real %s)\n(display (hidden))\n' "${symbol}" \
    >"${output}/negative.esk"
  if compile -L "${artifact}" --lib eshkol_transformer_e3_diagnostic \
      "${output}/negative.esk" -o "${output}/negative" \
      >"${output}/negative.log" 2>&1; then
    die "private E3 symbol escaped localization: ${symbol}"
  fi
  grep -F "${symbol}" "${output}/negative.log" >/dev/null || \
    die "private E3 link negative failed for another reason"
done
for entry in 'diagnostic-evaluate-fixed! 2' \
    'diagnostic-evaluation-f32-bits 2' 'diagnostic-evaluation-count 2'; do
  read -r operation arity <<<"${entry}"
  printf '(require transformer.evaluation)\n(%s)\n' "${operation}" \
    >"${output}/arity.esk"
  if compile --compile-only "${output}/arity.esk" -o "${output}/arity.o" \
      >"${output}/arity.log" 2>&1; then
    die "E3 public wrong arity accepted: ${operation}"
  fi
  grep -F "Arity mismatch: ${operation} expects ${arity} arguments but got 0" \
    "${output}/arity.log" >/dev/null || die "E3 arity negative failed for another reason"
done
printf 'E3 installed diagnostic package/facade gate passed: %s\n' "${output}"
