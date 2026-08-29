#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command timeout
require_command env
require_command nm
require_command rg
require_command strings
verify_toolchain

p1_runner="$(eshkol_build_dir)/eshkol-run"
p1_timeout="${P1_COMPILER_TIMEOUT_SECONDS:-180}"
[[ "${p1_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "P1_COMPILER_TIMEOUT_SECONDS must be a positive integer"

p1_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-p1.XXXXXX")"
trap 'rm -rf -- "${p1_tmp}"' EXIT

run_compiler() {
  timeout --foreground --signal=TERM --kill-after=5s \
    "${p1_timeout}s" env -u ESHKOL_PATH "${p1_runner}" "$@"
}

compile_production_object() {
  local source="$1" output="$2" depfile="$3" log="$4"
  run_compiler --strict-types --emit-object --no-stdlib \
    --emit-depfile "${depfile}" -I "${PROJECT_ROOT}/lib" \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -s "${output}" ]] || die "P1 compiler emitted no object: ${output}"
  [[ -s "${depfile}" ]] || die "P1 compiler emitted no depfile: ${depfile}"
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "P1 compiler reported ERROR while returning success: ${source}"
}

compile_provider_test_object() {
  local source="$1" output="$2" depfile="$3" log="$4"
  run_compiler --strict-types --emit-object --no-stdlib \
    --emit-depfile "${depfile}" -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/tests/p1/providers" \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -s "${output}" ]] || die "P1 compiler emitted no object: ${output}"
  [[ -s "${depfile}" ]] || die "P1 compiler emitted no depfile: ${depfile}"
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "P1 compiler reported ERROR while returning success: ${source}"
}

compile_production_aot() {
  local source="$1" output="$2" log="$3"
  run_compiler --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -x "${output}" ]] || die "P1 compiler emitted no executable: ${output}"
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "P1 AOT compiler reported ERROR while returning success: ${source}"
}

compile_provider_test_aot() {
  local source="$1" output="$2" log="$3"
  run_compiler --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/tests/p1/providers" \
    "${source}" -o "${output}" >"${log}" 2>&1
  [[ -x "${output}" ]] || die "P1 compiler emitted no executable: ${output}"
  ! grep -F 'ERROR:' "${log}" >/dev/null || \
    die "P1 AOT compiler reported ERROR while returning success: ${source}"
}

module_source="${PROJECT_ROOT}/lib/transformer/module.esk"
module_internal_source="${PROJECT_ROOT}/lib/transformer/module_internal.esk"
test_source="${PROJECT_ROOT}/tests/p1/module_state_test.esk"
provider_source="${PROJECT_ROOT}/tests/p1/providers/p1_test/tensor_provider.esk"
example_source="${PROJECT_ROOT}/examples/p1/module_state_inspection.esk"

for run in 1 2; do
  compile_production_object "${module_source}" "${p1_tmp}/module-${run}.o" \
    "${p1_tmp}/module-${run}.d" "${p1_tmp}/module-${run}.compile.log"
  compile_provider_test_object "${test_source}" "${p1_tmp}/test-${run}.o" \
    "${p1_tmp}/test-${run}.d" "${p1_tmp}/test-${run}.compile.log"
  compile_production_object "${example_source}" "${p1_tmp}/example-${run}.o" \
    "${p1_tmp}/example-${run}.d" "${p1_tmp}/example-${run}.compile.log"

  compile_provider_test_aot "${test_source}" "${p1_tmp}/test-${run}" \
    "${p1_tmp}/test-${run}.aot.log"
  timeout --foreground --signal=TERM --kill-after=2s 20s \
    "${p1_tmp}/test-${run}" >"${p1_tmp}/test-${run}.stdout" \
    2>"${p1_tmp}/test-${run}.stderr"

  compile_production_aot "${example_source}" "${p1_tmp}/example-${run}" \
    "${p1_tmp}/example-${run}.aot.log"
  timeout --foreground --signal=TERM --kill-after=2s 20s \
    "${p1_tmp}/example-${run}" >"${p1_tmp}/example-${run}.stdout" \
    2>"${p1_tmp}/example-${run}.stderr"
done

for run in 1 2; do
  for stem in module example; do
    depfile="${p1_tmp}/${stem}-${run}.d"
    grep -F "${module_source}" "${depfile}" >/dev/null || \
      die "P1 production depfile omits transformer.module: ${depfile}"
    grep -F "${module_internal_source}" "${depfile}" >/dev/null || \
      die "P1 production depfile omits transformer.module_internal: ${depfile}"
    if grep -F "${PROJECT_ROOT}/tests/" "${depfile}" >/dev/null; then
      die "P1 production dependency graph includes test support: ${depfile}"
    fi
  done

  grep -F "${provider_source}" "${p1_tmp}/test-${run}.d" >/dev/null || \
    die "P1 structural test omitted the explicit test-only provider"

  for stem in module example; do
    LC_ALL=C nm -a --format=posix "${p1_tmp}/${stem}-${run}.o" \
      >"${p1_tmp}/${stem}-${run}.nm"
    if rg -ni 'fixture' "${p1_tmp}/${stem}-${run}.nm"; then
      die "P1 production object contains a fixture symbol: ${stem}-${run}.o"
    fi
    if strings -a "${p1_tmp}/${stem}-${run}.o" | rg -ni 'fixture'; then
      die "P1 production object contains a fixture identifier: ${stem}-${run}.o"
    fi
  done
done

if ! LC_ALL=C nm -a --format=posix "${p1_tmp}/test-1.o" | rg -ni 'fixture' \
    >/dev/null; then
  die "P1 test object does not evidence the explicitly composed fixture provider"
fi

for stem in module test example; do
  cmp --silent "${p1_tmp}/${stem}-1.o" "${p1_tmp}/${stem}-2.o" || \
    die "P1 repeated ${stem} objects differ"
  cmp --silent "${p1_tmp}/${stem}-1.compile.log" \
    "${p1_tmp}/${stem}-2.compile.log" || \
    die "P1 repeated ${stem} compile diagnostics differ"
done

cmp --silent "${p1_tmp}/test-1.aot.log" "${p1_tmp}/test-2.aot.log" || \
  die "P1 repeated test AOT diagnostics differ"
cmp --silent "${p1_tmp}/example-1.aot.log" "${p1_tmp}/example-2.aot.log" || \
  die "P1 repeated example AOT diagnostics differ"
cmp --silent "${p1_tmp}/test-1.stdout" "${p1_tmp}/test-2.stdout" || \
  die "P1 repeated test output differs"
cmp --silent "${p1_tmp}/example-1.stdout" "${p1_tmp}/example-2.stdout" || \
  die "P1 repeated example output differs"
cmp --silent "${PROJECT_ROOT}/tests/expected/p1.stdout" \
  "${p1_tmp}/test-1.stdout" || die "P1 test output mismatch"
cmp --silent "${PROJECT_ROOT}/tests/expected/p1-example.stdout" \
  "${p1_tmp}/example-1.stdout" || die "P1 example output mismatch"
[[ ! -s "${p1_tmp}/test-1.stderr" && ! -s "${p1_tmp}/test-2.stderr" ]] || \
  die "P1 test executable wrote stderr"
[[ ! -s "${p1_tmp}/example-1.stderr" && ! -s "${p1_tmp}/example-2.stderr" ]] || \
  die "P1 example executable wrote stderr"

for run in 1 2; do
  negative_object="${p1_tmp}/negative-arity-${run}.o"
  negative_log="${p1_tmp}/negative-arity-${run}.log"
  if run_compiler --strict-types --emit-object --no-stdlib \
      -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/p1/negative_wrong_arity.esk" \
      -o "${negative_object}" >"${negative_log}" 2>&1; then
    die "P1 wrong-arity fixture unexpectedly compiled"
  fi
  grep -F 'Arity mismatch: module-state-dict expects 1 arguments but got 0' \
    "${negative_log}" >/dev/null || die "P1 wrong-arity diagnostic changed"
  [[ ! -e "${negative_object}" ]] || die "P1 negative compile left an object"

  negative_object="${p1_tmp}/negative-provider-${run}.o"
  negative_log="${p1_tmp}/negative-provider-${run}.log"
  if run_compiler --strict-types --emit-object --no-stdlib \
      -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/p1/negative_fixture_not_public.esk" \
      -o "${negative_object}" >"${negative_log}" 2>&1; then
    die "P1 test-only tensor provider unexpectedly compiled as public support"
  fi
  grep -F 'Unknown function: fixture-tensor-internal' "${negative_log}" \
    >/dev/null || die "P1 fixture-provider isolation diagnostic changed"
  [[ ! -e "${negative_object}" ]] || \
    die "P1 fixture-provider negative compile left an object"

  negative_object="${p1_tmp}/negative-provider-arity-${run}.o"
  negative_log="${p1_tmp}/negative-provider-arity-${run}.log"
  if run_compiler --strict-types --emit-object --no-stdlib \
      -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/p1/negative_provider_callback_arity.esk" \
      -o "${negative_object}" >"${negative_log}" 2>&1; then
    die "P1 wrong-arity tensor-provider callback unexpectedly compiled"
  fi
  grep -F "argument 2 of 'tensor-provider-create-internal'" "${negative_log}" \
    >/dev/null || die "P1 tensor-provider callback diagnostic changed"
  [[ ! -e "${negative_object}" ]] || \
    die "P1 tensor-provider callback negative compile left an object"
done
cmp --silent "${p1_tmp}/negative-arity-1.log" \
  "${p1_tmp}/negative-arity-2.log" || \
  die "P1 repeated negative diagnostics differ"
cmp --silent "${p1_tmp}/negative-provider-1.log" \
  "${p1_tmp}/negative-provider-2.log" || \
  die "P1 repeated provider-isolation diagnostics differ"
cmp --silent "${p1_tmp}/negative-provider-arity-1.log" \
  "${p1_tmp}/negative-provider-arity-2.log" || \
  die "P1 repeated provider-callback diagnostics differ"

if rg -ni '(fixture|tests/p1/providers|python|pytorch|torch\.load|finite[- ]difference|scalar[- ]fallback|cpu[- ]fallback)' \
    "${PROJECT_ROOT}/lib/transformer/module.esk" \
    "${PROJECT_ROOT}/lib/transformer/module_internal.esk"; then
  die "P1 production module contains a forbidden runtime/fallback reference"
fi

printf 'P1 PASS: strict compile, AOT, negatives, ownership, atomicity, and determinism\n'
