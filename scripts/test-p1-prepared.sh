#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar cmp grep python3 timeout; do require_command "${command}"; done
verify_toolchain

runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
compiler_timeout="${P1_PREPARED_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "P1_PREPARED_COMPILER_TIMEOUT_SECONDS must be a positive integer"

python3 "${PROJECT_ROOT}/scripts/check-p1-prepared-split.py"
"${PROJECT_ROOT}/scripts/generate-p1-roots.sh" --check

expected_closure=(
  internal/p1/lib/transformer/module.esk
  native/g3c4_p1_construction_extension.esk
  tests/p1/providers/p1_test/tensor_provider.esk
  tests/p1/prepared_construction_test.esk
)
mapfile -t actual_closure < \
  "${PROJECT_ROOT}/native/g3c4_p1_construction_source_closure.txt"
[[ "${actual_closure[*]}" == "${expected_closure[*]}" ]] || \
  die "P1 prepared construction source closure changed"

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-p1-prepared.XXXXXX")"
cleanup() {
  if [[ "${P1_PREPARED_KEEP_TMP:-0}" == 1 ]]; then
    printf 'P1 prepared evidence preserved: %s\n' "${temporary_dir}" >&2
  else
    rm -rf -- "${temporary_dir}"
  fi
}
trap cleanup EXIT

native_flags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1
  -I "${PROJECT_ROOT}/native"
)
"${cc}" "${native_flags[@]}" -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${temporary_dir}/p1_identity.o"
ar rcsD "${temporary_dir}/libeshkol_transformer_p1_identity.a" \
  "${temporary_dir}/p1_identity.o"

compile() {
  local output=$1 log=$2
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${temporary_dir}/cache-${output##*/}" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${compiler_timeout}s" "${runner}" --strict-types --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
      -I "${PROJECT_ROOT}/tests/p1/providers" \
      -L "${temporary_dir}" --lib eshkol_transformer_p1_identity \
      "${PROJECT_ROOT}/tests/p1/prepared_construction_test.esk" \
      -o "${output}" >"${log}" 2>&1
  [[ -x "${output}" ]] || die "P1 prepared AOT executable is missing"
}

for run in a b; do
  compile "${temporary_dir}/prepared-${run}" \
    "${temporary_dir}/compile-${run}.log"
  ESHKOL_ARENA_POISON=1 "${temporary_dir}/prepared-${run}" \
    >"${temporary_dir}/run-${run}.stdout"
done
cmp "${temporary_dir}/compile-a.log" "${temporary_dir}/compile-b.log"
cmp "${temporary_dir}/run-a.stdout" "${temporary_dir}/run-b.stdout"
grep -E '^P1 PREPARED PASS: checks=[1-9][0-9]*$' \
  "${temporary_dir}/run-a.stdout" >/dev/null

set +e
ESHKOL_ARENA_POISON=1 "${temporary_dir}/prepared-a" fatal \
  >"${temporary_dir}/fatal.stdout" 2>"${temporary_dir}/fatal.stderr"
fatal_status=$?
set -e
[[ "${fatal_status}" == 134 ]] || \
  die "prepared commit failure did not fail-stop with status 134: ${fatal_status}"

printf 'P1 PREPARED PASS: repeated strict AOT/runtime, prepared abort/seal, canonical eval, failure cleanup, fail-stop 134\n'
