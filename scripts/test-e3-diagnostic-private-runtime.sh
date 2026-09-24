#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in awk env grep python3 timeout; do require_command "${command}"; done

artifact="${1:-$(project_build_dir)/e3-diagnostic-private}"
output="${2:-$(project_build_dir)/e3-diagnostic-private-runtime}"
[[ -s "${artifact}/libeshkol_transformer_e3_diagnostic_private.a" ]] || \
  die "E3 diagnostic private archive prerequisite is missing"
mkdir -p "${output}"

corpus="${output}/corpus"
if [[ ! -s "${corpus}/corpus-reference.json" ]]; then
  python3 -m tests.e3_reference.corpus --output "${corpus}"
fi
reference="${E3_REFERENCE_BUNDLE:-}"
[[ "${reference}" == /* && -f "${reference}" ]] || \
  die "E3_REFERENCE_BUNDLE must name the accepted absolute reference bundle"
python3 - "${corpus}/corpus-reference.json" "${reference}" <<'PY'
from pathlib import Path
import sys
from tests.e3_reference.transcript import decode

corpus = decode(Path(sys.argv[1]).read_bytes())
reference = decode(Path(sys.argv[2]).read_bytes())
expected = {case["name"]: case for case in corpus["cases"]}
observed = {case["name"]: case["corpus"] for case in reference["cases"]}
if expected != observed:
    raise SystemExit("accepted E3 reference does not bind the generated corpus")
packed = next(case for case in reference["cases"]
              if case["name"] == "packed-single")
if (packed["synthetic"]["final"]["metrics_f32"] !=
        ["40b16271", "40a00000", "437f82e7", "00000000"] or
        packed["synthetic"]["final"]["counters"] != [5, 3]):
    raise SystemExit("diagnostic runtime expectations differ from accepted E3 reference")
PY

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E3 diagnostic compiler timeout must be a positive integer"

env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
  -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
  -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
  -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${output}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
  "${runner}" --strict-types --no-stdlib -O 2 \
    -L "${artifact}" --lib eshkol_transformer_e3_diagnostic_private \
    "${PROJECT_ROOT}/tests/e3_diagnostic_private/runtime.esk" \
    -o "${output}/runtime" \
    >"${output}/compile.stdout" 2>"${output}/compile.stderr"

for mode in success eos success-1024 success-2048 success-4096 success-8192 \
    failure-1024 failure-8192; do
  runtime_timeout=3600
  case "${mode}" in
    success-8192|failure-8192) runtime_timeout=7200 ;;
  esac
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    "${runtime_timeout}s" \
    "${output}/runtime" "${mode}" "${corpus}/packed-single" \
    >"${output}/${mode}.stdout" 2>"${output}/${mode}.stderr"
done
grep -Fx "E3-DIAGNOSTIC-SUCCESS-PASS" "${output}/success.stdout" >/dev/null
grep -Fx "E3-DIAGNOSTIC-EOS-PASS" "${output}/eos.stdout" >/dev/null
for mode in success-1024 success-2048 success-4096 success-8192 \
    failure-1024 failure-8192; do
  horizon=${mode##*-}
  kind=${mode%%-*}
  successes=${horizon}
  [[ "${kind}" == success ]] || successes=0
  grep -Ex "E3-DIAGNOSTIC-HORIZON-PASS horizon=${horizon} kind=${kind} arena_delta=[0-9]+ reservations=${horizon} successes=${successes} reachable_authority_bytes=[0-9]+ unreachable_staging_bytes=[0-9]+ inherited_transaction_bytes=[0-9]+ witness_other_bytes=[0-9]+ reserve_promoted_bytes=[0-9]+ envelope_bytes_per_call=[0-9]+ calibration_bytes=[0-9]+" \
    "${output}/${mode}.stdout" >/dev/null
done

printf 'kind\thorizon\tarena_delta\treachable_authority_bytes\tunreachable_staging_bytes\tinherited_transaction_bytes\twitness_other_bytes\treserve_promoted_bytes\tenvelope_bytes_per_call\tcalibration_bytes\n' \
  >"${output}/retention.tsv"
for mode in success-1024 success-2048 success-4096 success-8192 \
    failure-1024 failure-8192; do
  awk -v mode="${mode}" '
    /E3-DIAGNOSTIC-HORIZON-PASS/ {
      split(mode, parts, "-")
      for (i = 1; i <= NF; ++i) {
        split($i, value, "=")
        if ($i ~ /^arena_delta=/) { delta = value[2] }
        if ($i ~ /^reachable_authority_bytes=/) { authority = value[2] }
        if ($i ~ /^unreachable_staging_bytes=/) { staging = value[2] }
        if ($i ~ /^inherited_transaction_bytes=/) { inherited = value[2] }
        if ($i ~ /^witness_other_bytes=/) { witness = value[2] }
        if ($i ~ /^reserve_promoted_bytes=/) { promoted = value[2] }
        if ($i ~ /^envelope_bytes_per_call=/) { envelope = value[2] }
        if ($i ~ /^calibration_bytes=/) { calibration = value[2] }
      }
      print parts[1] "\t" parts[2] "\t" delta "\t" authority "\t" \
            staging "\t" inherited "\t" witness "\t" promoted "\t" envelope \
            "\t" calibration
    }
  ' "${output}/${mode}.stdout" >>"${output}/retention.tsv"
done

printf 'E3 diagnostic private runtime planned gate passed: %s\n' \
  "${output}/retention.tsv"
