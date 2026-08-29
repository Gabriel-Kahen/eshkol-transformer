#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command timeout
require_command cmp
require_command grep
verify_toolchain

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
[[ "$(tsv_value "${provenance}" eshkol_repository)" == \
   "https://github.com/tsotchke/eshkol.git" ]] || \
  die "T1 requires canonical tsotchke/eshkol provenance"
[[ "$(lock_value eshkol_commit)" == \
   "90cbd7130f47b8184bcc77b8d5c1b0026da980de" ]] || \
  die "T1 toolchain lock does not contain the accepted canonical commit"
[[ "$(lock_value eshkol_version)" == "1.3.4-evolve" ]] || \
  die "T1 toolchain lock does not contain compiler v1.3.4-evolve"

python_exec="${T1_PYTHON:-python3}"
require_command "${python_exec}"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-t1.XXXXXX")"
trap 'rm -rf -- "${test_root}"' EXIT

(
  cd -- "${PROJECT_ROOT}"
  PYTHONDONTWRITEBYTECODE=1 "${python_exec}" -m unittest discover \
    -s tests/t1 -p 'test_*.py' -v
)

for generated in first second; do
  (
    cd -- "${PROJECT_ROOT}"
    PYTHONDONTWRITEBYTECODE=1 "${python_exec}" -m tests.t1.generate_fixture \
      --output "${test_root}/${generated}.tsv"
  )
done
cmp --silent "${test_root}/first.tsv" "${test_root}/second.tsv" || \
  die "fresh T1 fixture generations differ"
cmp --silent "${test_root}/first.tsv" \
  "${PROJECT_ROOT}/tests/t1/fixtures/byte_tokenizer_v1.tsv" || \
  die "fresh T1 fixture differs from the frozen fixture"

compiler="$(eshkol_build_dir)/eshkol-run"
cxx="$(tsv_value "${provenance}" cxx_path)"
compile_timeout="${T1_COMPILE_TIMEOUT_SECONDS:-180}"
run_timeout="${T1_RUN_TIMEOUT_SECONDS:-15}"
[[ "${compile_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "T1_COMPILE_TIMEOUT_SECONDS must be a positive integer"
[[ "${run_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "T1_RUN_TIMEOUT_SECONDS must be a positive integer"

for probe in bytevector_capability i64_storage_blocker; do
  for repetition in 1 2; do
    binary="${test_root}/${probe}-${repetition}"
    compile_stdout="${test_root}/${probe}-${repetition}.compile.stdout"
    compile_stderr="${test_root}/${probe}-${repetition}.compile.stderr"
    ESHKOL_CXX_COMPILER="${cxx}" timeout --foreground --signal=TERM --kill-after=5s \
      "${compile_timeout}s" "${compiler}" --no-stdlib \
      "${PROJECT_ROOT}/tests/t1/probes/${probe}.esk" -o "${binary}" \
      >"${compile_stdout}" 2>"${compile_stderr}"
    [[ -x "${binary}" ]] || die "T1 compiler emitted no executable for ${probe}"
    if grep -E -i 'ERROR:|Failed to|terminate called|Assertion' \
        "${compile_stdout}" "${compile_stderr}" >/dev/null; then
      die "T1 compilation emitted an error diagnostic for ${probe}"
    fi
    (
      cd -- "${test_root}"
      timeout --foreground --signal=TERM --kill-after=5s "${run_timeout}s" \
        "${binary}"
    ) >"${test_root}/${probe}-${repetition}.stdout" \
      2>"${test_root}/${probe}-${repetition}.stderr"
    if grep -E -i 'ERROR:|Failed to|terminate called|Assertion' \
        "${test_root}/${probe}-${repetition}.stdout" \
        "${test_root}/${probe}-${repetition}.stderr" >/dev/null; then
      die "T1 execution emitted an error diagnostic for ${probe}"
    fi
  done
  cmp --silent "${test_root}/${probe}-1.stdout" \
    "${test_root}/${probe}-2.stdout" || \
    die "fresh AOT executions differ for ${probe}"
done

grep -Fx 'T1_DEV_BYTEVECTOR PASS all 256 values round trip' \
  "${test_root}/bytevector_capability-1.stdout" >/dev/null
grep -Fx 'T1_PUBLIC_TOKENIZER_RUNTIME BLOCKED' \
  "${test_root}/bytevector_capability-1.stdout" >/dev/null
grep -Fx \
  'T1_I64_STORAGE BLOCKED integer-values-exposed-but-storage-dtype-unobservable' \
  "${test_root}/i64_storage_blocker-1.stdout" >/dev/null

printf 'T1 development format: PASS (strict deterministic TSV v1.0 fixture)\n'
printf 'T1 bytevector capability: PASS (all 256 values, two fresh AOT runs)\n'
printf 'T1 public tokenizer runtime: BLOCKED (contiguous i64 storage unverified)\n'
