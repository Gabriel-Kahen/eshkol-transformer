#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command timeout
require_command python3
require_command ar
require_command nm
require_command strings
verify_toolchain

D1_TIMEOUT=$(command -v timeout)
D1_PYTHON=$(command -v python3)
D1_RUNNER="$(eshkol_build_dir)/eshkol-run"
D1_CXX="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
D1_TIMEOUT_SECONDS=${D1_TIMEOUT_SECONDS:-60}
D1_ARTIFACT_DIR="$(project_build_dir)/d1"
D1_LIBRARY="${D1_ARTIFACT_DIR}/libeshkol_transformer_d1.a"
[[ "${D1_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
  die "D1_TIMEOUT_SECONDS must be a positive integer"
[[ -r "${D1_LIBRARY}" ]] || die "canonical D1 native archive is missing"
[[ -r "${D1_ARTIFACT_DIR}/data_io.o" ]] || \
  die "canonical D1 native object is missing"
[[ "$(ar t "${D1_LIBRARY}")" == "data_io.o" ]] || \
  die "canonical D1 native archive has unexpected members"

D1_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-d1.XXXXXX")
trap 'rm -rf -- "${D1_TMP}"' EXIT

d1_compile() {
  local source=$1 output=$2
  local cache_home=${D1_CACHE_HOME:-${D1_TMP}/cache}
  local library_dir=${D1_LIBRARY_DIR:-${D1_ARTIFACT_DIR}}
  local library_name=${D1_LIBRARY_NAME:-eshkol_transformer_d1}
  shift 2
  mkdir -p "${cache_home}"
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
    env XDG_CACHE_HOME="${cache_home}" ESHKOL_CXX_COMPILER="${D1_CXX}" \
    "${D1_RUNNER}" \
    --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/src" \
    -L "${library_dir}" --lib "${library_name}" \
    "$@" "${source}" -o "${output}"
}

D1_FAULT_ARTIFACT_DIR="${D1_TMP}/fault-artifacts"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-d1.sh" \
  "${D1_FAULT_ARTIFACT_DIR}" test-faults \
  >"${D1_TMP}/fault-build.stdout" 2>"${D1_TMP}/fault-build.stderr"
D1_FAULT_LIBRARY="${D1_FAULT_ARTIFACT_DIR}/libeshkol_transformer_d1_faults.a"
[[ -r "${D1_FAULT_LIBRARY}" ]] || die "D1 fault-test archive is missing"
[[ "$(ar t "${D1_FAULT_LIBRARY}")" == "data_io.o" ]] || \
  die "D1 fault-test archive has unexpected members"
for forbidden_fault_text in ET_D1_TEST_FAULT ET_D1_TEST_FAIL_CALL \
    short-write write-enospc write-eio close-eio; do
  if strings "${D1_LIBRARY}" | grep -F "${forbidden_fault_text}" >/dev/null; then
    die "canonical D1 native archive contains test-fault control: ${forbidden_fault_text}"
  fi
done
if nm -u "${D1_LIBRARY}" | grep -E ' (getenv|strtoull|strcmp)$' >/dev/null; then
  die "canonical D1 native archive references test-fault runtime helpers"
fi

for run in 1 2; do
  d1_compile "${PROJECT_ROOT}/tests/d1/compile_data_api.esk" \
    "${D1_TMP}/data-api-${run}.o" \
    --emit-depfile "${D1_TMP}/data-api-${run}.d" --compile-only \
    >"${D1_TMP}/data-api-${run}.stdout" 2>"${D1_TMP}/data-api-${run}.stderr"
  test -s "${D1_TMP}/data-api-${run}.o"
  grep -F 'lib/transformer/data.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'src/eshkol_transformer/token_shard.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'src/eshkol_transformer/sha256.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'lib/transformer/error_public.esk' \
    "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'lib/transformer/error_core.esk' \
    "${D1_TMP}/data-api-${run}.d" >/dev/null
  if grep -F 'lib/transformer/error_internal.esk' \
      "${D1_TMP}/data-api-${run}.d" >/dev/null; then
    die "D1 public data graph unexpectedly includes transformer.error_internal"
  fi
done
cmp --silent "${D1_TMP}/data-api-1.o" "${D1_TMP}/data-api-2.o"
cmp --silent "${D1_TMP}/data-api-1.stdout" "${D1_TMP}/data-api-2.stdout"
for compile_log in "${D1_TMP}"/data-api-*.stdout "${D1_TMP}"/data-api-*.stderr; do
  if grep -F 'ERROR:' "${compile_log}" >/dev/null; then
    die "D1 compile-only gate emitted ERROR in ${compile_log}"
  fi
done

hidden_error_names=(
  transformer-error-make
  transformer-error-raise
  transformer-error-wrap-foreign
)
hidden_error_sources=(make raise wrap)
for hidden_index in "${!hidden_error_names[@]}"; do
  hidden_name="${hidden_error_names[${hidden_index}]}"
  hidden_source="${hidden_error_sources[${hidden_index}]}"
  for run in 1 2; do
    if D1_CACHE_HOME="${D1_TMP}/negative-error-${hidden_source}-cache-${run}" \
        d1_compile \
        "${PROJECT_ROOT}/tests/d1/negative_error_${hidden_source}_public.esk" \
        "${D1_TMP}/negative-error-${hidden_source}-${run}.o" \
        --compile-only \
        >"${D1_TMP}/negative-error-${hidden_source}-${run}.log" 2>&1; then
      die "D1 hidden E1 ${hidden_source} helper unexpectedly compiled publicly"
    fi
    grep -F "Unbound variable: ${hidden_name}" \
      "${D1_TMP}/negative-error-${hidden_source}-${run}.log" >/dev/null
    test ! -e "${D1_TMP}/negative-error-${hidden_source}-${run}.o"
  done
  cmp --silent "${D1_TMP}/negative-error-${hidden_source}-1.log" \
    "${D1_TMP}/negative-error-${hidden_source}-2.log"
done

hidden_summary_names=(d1-token-summary-create d1-token-make-summary)
hidden_summary_sources=(constructor maker)
for hidden_index in "${!hidden_summary_names[@]}"; do
  hidden_name="${hidden_summary_names[${hidden_index}]}"
  hidden_source="${hidden_summary_sources[${hidden_index}]}"
  for run in 1 2; do
    if d1_compile \
        "${PROJECT_ROOT}/tests/d1/negative_summary_${hidden_source}_public.esk" \
        "${D1_TMP}/negative-summary-${hidden_source}-${run}.o" \
        --compile-only \
        >"${D1_TMP}/negative-summary-${hidden_source}-${run}.log" 2>&1; then
      die "D1 hidden summary ${hidden_source} unexpectedly compiled publicly"
    fi
    grep -F "Unbound variable: ${hidden_name}" \
      "${D1_TMP}/negative-summary-${hidden_source}-${run}.log" >/dev/null
    test ! -e "${D1_TMP}/negative-summary-${hidden_source}-${run}.o"
  done
  cmp --silent "${D1_TMP}/negative-summary-${hidden_source}-1.log" \
    "${D1_TMP}/negative-summary-${hidden_source}-2.log"
done

d1_compile "${PROJECT_ROOT}/tests/d1/sha256_probe.esk" \
  "${D1_TMP}/sha256-probe" \
  >"${D1_TMP}/sha256.compile.stdout" 2>"${D1_TMP}/sha256.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/primitive_probe.esk" \
  "${D1_TMP}/primitive-probe" \
  >"${D1_TMP}/primitive.compile.stdout" 2>"${D1_TMP}/primitive.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${D1_TMP}/corpus-tool" \
  --emit-depfile "${D1_TMP}/corpus-tool.d" \
  >"${D1_TMP}/corpus.compile.stdout" 2>"${D1_TMP}/corpus.compile.stderr"
D1_LIBRARY_DIR="${D1_FAULT_ARTIFACT_DIR}" \
D1_LIBRARY_NAME=eshkol_transformer_d1_faults \
D1_CACHE_HOME="${D1_TMP}/fault-tool-cache" \
  d1_compile "${PROJECT_ROOT}/tests/d1/corpus_tool.esk" \
  "${D1_TMP}/corpus-tool-faults" \
  >"${D1_TMP}/corpus-faults.compile.stdout" \
  2>"${D1_TMP}/corpus-faults.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/arithmetic_probe.esk" \
  "${D1_TMP}/arithmetic-probe" \
  >"${D1_TMP}/arithmetic.compile.stdout" 2>"${D1_TMP}/arithmetic.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/e1_mapping_probe.esk" \
  "${D1_TMP}/e1-mapping-probe" \
  >"${D1_TMP}/e1-mapping.compile.stdout" 2>"${D1_TMP}/e1-mapping.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/summary_opacity_probe.esk" \
  "${D1_TMP}/summary-opacity-probe" \
  >"${D1_TMP}/summary-opacity.compile.stdout" \
  2>"${D1_TMP}/summary-opacity.compile.stderr"

for compile_log in "${D1_TMP}"/*.compile.stdout "${D1_TMP}"/*.compile.stderr; do
  if grep -F 'ERROR:' "${compile_log}" >/dev/null; then
    die "D1 compiler emitted ERROR in ${compile_log}"
  fi
done

mkdir "${D1_TMP}/primitive-root"
PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/sha256-probe" >"${D1_TMP}/sha256.stdout" 2>"${D1_TMP}/sha256.stderr"
grep -Fx 'D1_SHA256_PASS' "${D1_TMP}/sha256.stdout" >/dev/null
test ! -s "${D1_TMP}/sha256.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/primitive-probe" "${D1_TMP}/primitive-root" \
  >"${D1_TMP}/primitive.stdout" 2>"${D1_TMP}/primitive.stderr"
grep -Fx 'D1_PRIMITIVES_PASS' "${D1_TMP}/primitive.stdout" >/dev/null
test ! -s "${D1_TMP}/primitive.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/arithmetic-probe" >"${D1_TMP}/arithmetic.stdout" \
  2>"${D1_TMP}/arithmetic.stderr"
grep -Fx 'D1_ARITHMETIC_PASS' "${D1_TMP}/arithmetic.stdout" >/dev/null
test ! -s "${D1_TMP}/arithmetic.stderr"

PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/e1-mapping-probe" >"${D1_TMP}/e1-mapping.stdout" \
  2>"${D1_TMP}/e1-mapping.stderr"
grep -Fx 'D1_E1_MAPPING_PASS' "${D1_TMP}/e1-mapping.stdout" >/dev/null
test ! -s "${D1_TMP}/e1-mapping.stderr"

mkdir "${D1_TMP}/summary-writer" "${D1_TMP}/summary-validator"
PATH=/definitely-not-a-real-path \
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
  "${D1_TMP}/summary-opacity-probe" \
  "${D1_TMP}/summary-writer" "${D1_TMP}/summary-validator" \
  >"${D1_TMP}/summary-opacity.stdout" \
  2>"${D1_TMP}/summary-opacity.stderr"
grep -Fx 'D1_SUMMARY_OPACITY_PASS' "${D1_TMP}/summary-opacity.stdout" >/dev/null
test ! -s "${D1_TMP}/summary-opacity.stderr"

PYTHONDONTWRITEBYTECODE=1 D1_CORPUS_TOOL="${D1_TMP}/corpus-tool" \
  D1_FAULT_CORPUS_TOOL="${D1_TMP}/corpus-tool-faults" \
  "${D1_PYTHON}" -m unittest discover -s "${PROJECT_ROOT}/tests/d1" \
  -p 'test_*.py' -v

if find "${PROJECT_ROOT}/src" -type f \( -name '*.py' -o -name '*.pyc' \) -print -quit | \
    grep -q .; then
  die "D1 production source contains Python"
fi

printf 'D1 PASS: deterministic native writer, strict validator, primitive probes, and format tests\n'
