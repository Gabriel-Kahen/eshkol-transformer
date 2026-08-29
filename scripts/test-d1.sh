#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command timeout
require_command python3
verify_toolchain

D1_TIMEOUT=$(command -v timeout)
D1_PYTHON=$(command -v python3)
D1_RUNNER="$(eshkol_build_dir)/eshkol-run"
D1_CXX="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
D1_TIMEOUT_SECONDS=${D1_TIMEOUT_SECONDS:-60}
[[ "${D1_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
  die "D1_TIMEOUT_SECONDS must be a positive integer"

D1_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-d1.XXXXXX")
trap 'rm -rf -- "${D1_TMP}"' EXIT

d1_compile() {
  local source=$1 output=$2
  shift 2
  "${D1_TIMEOUT}" --foreground --signal=TERM --kill-after=5s "${D1_TIMEOUT_SECONDS}s" \
    env ESHKOL_CXX_COMPILER="${D1_CXX}" "${D1_RUNNER}" \
    --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/src" \
    "$@" "${source}" -o "${output}"
}

for run in 1 2; do
  d1_compile "${PROJECT_ROOT}/tests/d1/compile_data_api.esk" \
    "${D1_TMP}/data-api-${run}.o" \
    --emit-depfile "${D1_TMP}/data-api-${run}.d" --compile-only \
    >"${D1_TMP}/data-api-${run}.stdout" 2>"${D1_TMP}/data-api-${run}.stderr"
  test -s "${D1_TMP}/data-api-${run}.o"
  grep -F 'lib/transformer/data.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'src/eshkol_transformer/token_shard.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'src/eshkol_transformer/sha256.esk' "${D1_TMP}/data-api-${run}.d" >/dev/null
  grep -F 'lib/transformer/error_internal.esk' \
    "${D1_TMP}/data-api-${run}.d" >/dev/null
done
cmp --silent "${D1_TMP}/data-api-1.o" "${D1_TMP}/data-api-2.o"
cmp --silent "${D1_TMP}/data-api-1.stdout" "${D1_TMP}/data-api-2.stdout"
for compile_log in "${D1_TMP}"/data-api-*.stdout "${D1_TMP}"/data-api-*.stderr; do
  if grep -F 'ERROR:' "${compile_log}" >/dev/null; then
    die "D1 compile-only gate emitted ERROR in ${compile_log}"
  fi
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
d1_compile "${PROJECT_ROOT}/tests/d1/arithmetic_probe.esk" \
  "${D1_TMP}/arithmetic-probe" \
  >"${D1_TMP}/arithmetic.compile.stdout" 2>"${D1_TMP}/arithmetic.compile.stderr"
d1_compile "${PROJECT_ROOT}/tests/d1/e1_mapping_probe.esk" \
  "${D1_TMP}/e1-mapping-probe" \
  >"${D1_TMP}/e1-mapping.compile.stdout" 2>"${D1_TMP}/e1-mapping.compile.stderr"

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

PYTHONDONTWRITEBYTECODE=1 D1_CORPUS_TOOL="${D1_TMP}/corpus-tool" \
  "${D1_PYTHON}" -m unittest discover -s "${PROJECT_ROOT}/tests/d1" \
  -p 'test_*.py' -v

if find "${PROJECT_ROOT}/src" -type f \( -name '*.py' -o -name '*.pyc' \) -print -quit | \
    grep -q .; then
  die "D1 production source contains Python"
fi

printf 'D1 PASS: deterministic native writer, strict validator, primitive probes, and format tests\n'
