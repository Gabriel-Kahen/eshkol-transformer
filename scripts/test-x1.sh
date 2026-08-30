#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command cmp
require_command ldd
require_command python3
require_command sha256sum
require_command timeout
verify_toolchain

x1_runner="$(eshkol_build_dir)/eshkol-run"
x1_timeout=${X1_COMPILER_TIMEOUT_SECONDS:-60}
[[ "${x1_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "X1_COMPILER_TIMEOUT_SECONDS must be a positive integer"
x1_tmp=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-x1.XXXXXX")
trap 'rm -rf -- "${x1_tmp}"' EXIT

run_compiler() {
  timeout --foreground --signal=TERM --kill-after=5s "${x1_timeout}s" \
    "${x1_runner}" "$@"
}

compile_object() {
  local source=$1 output=$2 depfile=$3 log=$4
  run_compiler --strict-types --emit-object --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    --emit-depfile "${depfile}" "${source}" -o "${output}" \
    >"${log}" 2>&1
  ! grep -F 'ERROR:' "${log}" >/dev/null || die "Eshkol reported ERROR while compiling ${source}"
  [[ -s "${output}" ]] || die "Eshkol did not emit ${output}"
  [[ -s "${depfile}" ]] || die "Eshkol did not emit ${depfile}"
}

compile_binary() {
  local source=$1 output=$2 log=$3
  run_compiler --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    "${source}" -o "${output}" >"${log}" 2>&1
  ! grep -F 'ERROR:' "${log}" >/dev/null || die "Eshkol reported ERROR while compiling ${source}"
  [[ -x "${output}" ]] || die "Eshkol did not emit executable ${output}"
}

compile_negative_unknown() {
  local source=$1 output=$2 log=$3 symbol=$4
  if run_compiler --strict-types --emit-object --no-stdlib \
      -I "${PROJECT_ROOT}/lib" "${source}" -o "${output}" \
      >"${log}" 2>&1; then
    die "negative X1 fixture unexpectedly compiled: ${source}"
  fi
  [[ ! -e "${output}" ]] || die "negative X1 fixture emitted an artifact: ${source}"
  grep -F "Unknown function: ${symbol}" "${log}" >/dev/null || \
    die "negative X1 fixture did not fail on ${symbol}: ${source}"
}

for x1_run in 1 2; do
  compile_object \
    "${PROJECT_ROOT}/tests/x1/test_config.esk" \
    "${x1_tmp}/test-config-${x1_run}.o" \
    "${x1_tmp}/test-config-${x1_run}.d" \
    "${x1_tmp}/test-config-${x1_run}.compile.log"
done
cmp --silent "${x1_tmp}/test-config-1.o" "${x1_tmp}/test-config-2.o" || \
  die "fresh X1 objects differ"
grep -F 'lib/transformer/config.esk' "${x1_tmp}/test-config-1.d" >/dev/null || \
  die "X1 depfile does not prove production config module compilation"
grep -F 'lib/transformer/error_internal.esk' \
  "${x1_tmp}/test-config-1.d" >/dev/null || \
  die "X1 depfile does not prove production E1 module compilation"

compile_negative_unknown \
  "${PROJECT_ROOT}/tests/x1/negative_internal_constructor.esk" \
  "${x1_tmp}/negative-constructor.o" \
  "${x1_tmp}/negative-constructor.log" x1-make-resolved
compile_negative_unknown \
  "${PROJECT_ROOT}/tests/x1/negative_internal_accessor.esk" \
  "${x1_tmp}/negative-accessor.o" \
  "${x1_tmp}/negative-accessor.log" x1-resolved-values

compile_binary "${PROJECT_ROOT}/tests/x1/test_config.esk" \
  "${x1_tmp}/test-config" "${x1_tmp}/test-config.compile.log"
compile_binary "${PROJECT_ROOT}/tests/x1/emit_minimal.esk" \
  "${x1_tmp}/emit-minimal" "${x1_tmp}/emit-minimal.compile.log"
compile_binary "${PROJECT_ROOT}/tests/x1/emit_minimal_fingerprint.esk" \
  "${x1_tmp}/emit-fingerprint" "${x1_tmp}/emit-fingerprint.compile.log"

[[ ! -e "${PROJECT_ROOT}/should-not-run" && ! -e "${x1_tmp}/should-not-run" ]] || \
  die "hostile-input sentinel already exists"
if ! env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV='$(touch should-not-run)' \
  PYTHON=/definitely/unavailable PYTHONPATH=/definitely/unavailable \
  timeout --foreground --signal=TERM --kill-after=2s 30s \
  "${x1_tmp}/test-config" >"${x1_tmp}/test-1.stdout"; then
  sed -n '1,240p' "${x1_tmp}/test-1.stdout" >&2
  die "X1 native behavior test failed"
fi
(
  cd -- "${x1_tmp}"
  env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV='${HOME}:include:eval' \
    PYTHON=/another/unavailable PYTHONPATH=/another/unavailable \
    timeout --foreground --signal=TERM --kill-after=2s 30s \
    "${x1_tmp}/test-config" >"${x1_tmp}/test-2.stdout"
)
cmp --silent "${x1_tmp}/test-1.stdout" "${x1_tmp}/test-2.stdout" || \
  die "X1 output changed with CWD or hostile environment"
[[ ! -e "${PROJECT_ROOT}/should-not-run" && ! -e "${x1_tmp}/should-not-run" ]] || \
  die "executable-looking configuration input created a sentinel"
grep -Fx 'X1 SUMMARY: 110 passed, 0 failed' "${x1_tmp}/test-1.stdout" >/dev/null || \
  die "X1 native behavior summary did not report all tests passing"

env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=first \
  timeout --foreground --signal=TERM --kill-after=2s 30s \
  "${x1_tmp}/emit-minimal" >"${x1_tmp}/resolved-1.json"
env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=first \
  timeout --foreground --signal=TERM --kill-after=2s 30s \
  "${x1_tmp}/emit-fingerprint" >"${x1_tmp}/resolved-1.sha256"
(
  cd -- "${x1_tmp}"
  env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=second \
    timeout --foreground --signal=TERM --kill-after=2s 30s \
    "${x1_tmp}/emit-minimal" >"${x1_tmp}/resolved-2.json"
  env -i PATH=/usr/bin:/bin X1_HOSTILE_ENV=second \
    timeout --foreground --signal=TERM --kill-after=2s 30s \
    "${x1_tmp}/emit-fingerprint" >"${x1_tmp}/resolved-2.sha256"
)
cmp --silent "${x1_tmp}/resolved-1.json" "${x1_tmp}/resolved-2.json" || \
  die "fresh native canonical manifests differ"
cmp --silent "${x1_tmp}/resolved-1.sha256" "${x1_tmp}/resolved-2.sha256" || \
  die "fresh native fingerprints differ"
cmp --silent "${PROJECT_ROOT}/tests/x1/fixtures/resolved_minimal_v1.json" \
  "${x1_tmp}/resolved-1.json" || die "native canonical bytes differ from checked fixture"
cmp --silent "${PROJECT_ROOT}/tests/x1/fixtures/resolved_minimal_v1.sha256" \
  "${x1_tmp}/resolved-1.sha256" || die "native fingerprint differs from checked fixture"
x1_digest=$(sha256sum "${x1_tmp}/resolved-1.json" | awk '{print $1}')
grep -Fx "sha256:eshkol-config-json-v1:${x1_digest}" \
  "${x1_tmp}/resolved-1.sha256" >/dev/null || \
  die "native fingerprint does not cover exact canonical bytes"

if ldd "${x1_tmp}/test-config" | grep -Eiq 'python|torch'; then
  die "X1 native binary links a Python or Torch runtime"
fi

export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "${PROJECT_ROOT}/tests/x1" -p 'test_*.py' -v

printf 'X1 PASS: repeated strict compile, native behavior, canonical bytes, fingerprint, isolation\n'
