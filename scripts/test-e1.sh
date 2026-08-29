#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command timeout
verify_toolchain

e1_runner="$(eshkol_build_dir)/eshkol-run"
e1_cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
e1_timeout_seconds="${E1_COMPILER_TIMEOUT_SECONDS:-60}"
e1_runtime_timeout_seconds="${E1_RUNTIME_TIMEOUT_SECONDS:-10}"
[[ "${e1_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || \
  die "E1_COMPILER_TIMEOUT_SECONDS must be a positive integer"
[[ "${e1_runtime_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || \
  die "E1_RUNTIME_TIMEOUT_SECONDS must be a positive integer"

e1_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-e1.XXXXXX")"
trap 'rm -rf -- "${e1_tmp}"' EXIT
mkdir -p "${e1_tmp}/cache"

run_compiler() {
  XDG_CACHE_HOME="${e1_tmp}/cache" ESHKOL_CXX_COMPILER="${e1_cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${e1_timeout_seconds}s" "${e1_runner}" "$@"
}

run_e1_program() {
  timeout --foreground --signal=TERM --kill-after=5s \
    "${e1_runtime_timeout_seconds}s" "$@"
}

provide_words() {
  sed -n '/^(provide /,/)/p' "$1" | \
    sed -e '1s/^(provide //' -e '$s/)$//' | tr '\n' ' ' | xargs
}

e1_public_exports="transformer-error? transformer-error-category transformer-error-operation transformer-error-message transformer-error-details transformer-error-cause"
e1_internal_exports="${e1_public_exports} transformer-error-make transformer-error-raise transformer-error-wrap-foreign"
e1_core_exports="${e1_public_exports} e1-internal-dispatch"
[[ "$(provide_words "${PROJECT_ROOT}/lib/transformer/error_core.esk")" == \
    "${e1_core_exports}" ]] || die "E1 core declared exports drifted"
[[ "$(provide_words "${PROJECT_ROOT}/lib/transformer/error_public.esk")" == \
    "${e1_public_exports}" ]] || die "E1 public declared exports drifted"
[[ "$(provide_words "${PROJECT_ROOT}/lib/transformer/error_internal.esk")" == \
    "${e1_internal_exports}" ]] || die "E1 internal declared exports drifted"

compile_e1() {
  local source="$1" output="$2" depfile="$3"
  run_compiler --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    --emit-depfile "${depfile}" --compile-only \
    "${source}" -o "${output}"
}

for run in 1 2; do
  compile_e1 \
    "${PROJECT_ROOT}/tests/e1/error_contract.esk" \
    "${e1_tmp}/error-contract-${run}.o" \
    "${e1_tmp}/error-contract-${run}.d" \
    >"${e1_tmp}/compile-${run}.stdout" 2>"${e1_tmp}/compile-${run}.stderr"

  run_compiler --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    "${PROJECT_ROOT}/tests/e1/error_contract.esk" \
    -o "${e1_tmp}/error-contract-${run}" \
    >"${e1_tmp}/link-${run}.stdout" 2>"${e1_tmp}/link-${run}.stderr"
  run_e1_program "${e1_tmp}/error-contract-${run}" \
    >"${e1_tmp}/aot-${run}.stdout" 2>"${e1_tmp}/aot-${run}.stderr"

  run_compiler --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -r "${PROJECT_ROOT}/tests/e1/error_contract.esk" \
    >"${e1_tmp}/jit-${run}.stdout" 2>"${e1_tmp}/jit-${run}.stderr"

  if grep -F 'ERROR:' "${e1_tmp}/compile-${run}.stderr" \
      "${e1_tmp}/link-${run}.stderr" "${e1_tmp}/jit-${run}.stderr" >/dev/null; then
    die "E1 compiler reported ERROR during positive run ${run}"
  fi
done

cmp "${e1_tmp}/error-contract-1.o" "${e1_tmp}/error-contract-2.o"
cmp "${e1_tmp}/compile-1.stdout" "${e1_tmp}/compile-2.stdout"
cmp "${e1_tmp}/aot-1.stdout" "${e1_tmp}/aot-2.stdout"
cmp "${e1_tmp}/jit-1.stdout" "${e1_tmp}/jit-2.stdout"
cmp "${e1_tmp}/aot-1.stdout" "${e1_tmp}/jit-1.stdout"
cmp "${PROJECT_ROOT}/tests/expected/e1.stdout" "${e1_tmp}/aot-1.stdout"
test ! -s "${e1_tmp}/aot-1.stderr"
test ! -s "${e1_tmp}/aot-2.stderr"
grep -F 'lib/transformer/error_internal.esk' \
  "${e1_tmp}/error-contract-1.d" >/dev/null || \
  die "E1 depfile does not prove production error module compilation"

for run in 1 2; do
  run_compiler --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -r "${PROJECT_ROOT}/tests/e1/compile_error_internal.esk" \
    >"${e1_tmp}/internal-${run}.stdout" 2>"${e1_tmp}/internal-${run}.stderr"

  run_compiler --strict-types --no-stdlib \
    -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/tests/fixtures/a0" \
    -I "${PROJECT_ROOT}/tests/fixtures/e1" \
    -r "${PROJECT_ROOT}/tests/fixtures/e1/compile_public_errors.esk" \
    >"${e1_tmp}/public-${run}.stdout" 2>"${e1_tmp}/public-${run}.stderr"
done
cmp "${e1_tmp}/internal-1.stdout" "${e1_tmp}/internal-2.stdout"
grep -Fx 'e1-compile:v1' "${e1_tmp}/internal-1.stdout" >/dev/null
cmp "${e1_tmp}/public-1.stdout" "${e1_tmp}/public-2.stdout"
grep -Fx 'e1-public-errors:v1' "${e1_tmp}/public-1.stdout" >/dev/null

for run in 1 2; do
  if run_compiler --strict-types --no-stdlib --compile-only \
      -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/tests/fixtures/a0" \
      -I "${PROJECT_ROOT}/tests/fixtures/e1" \
      "${PROJECT_ROOT}/tests/fixtures/e1/negative_internal_public_export.esk" \
      -o "${e1_tmp}/public-leak-${run}.o" \
      >"${e1_tmp}/public-leak-${run}.log" 2>&1; then
    die "E1 internal helper unexpectedly compiled through transformer.public"
  fi
  grep -F 'Unbound variable: transformer-error-make' \
    "${e1_tmp}/public-leak-${run}.log" >/dev/null
  test ! -e "${e1_tmp}/public-leak-${run}.o"

  if run_compiler --strict-types --no-stdlib --compile-only \
      -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/tests/fixtures/a0" \
      -I "${PROJECT_ROOT}/tests/fixtures/e1" \
      "${PROJECT_ROOT}/tests/fixtures/e1/negative_internal_capabilities_export.esk" \
      -o "${e1_tmp}/capabilities-leak-${run}.o" \
      >"${e1_tmp}/capabilities-leak-${run}.log" 2>&1; then
    die "E1 internal helper unexpectedly compiled through transformer.capabilities"
  fi
  grep -F 'Unbound variable: transformer-error-make' \
    "${e1_tmp}/capabilities-leak-${run}.log" >/dev/null
  test ! -e "${e1_tmp}/capabilities-leak-${run}.o"

  if run_compiler --strict-types --no-stdlib --compile-only \
      -I "${PROJECT_ROOT}/lib" \
      "${PROJECT_ROOT}/tests/fixtures/e1/negative_accessor_wrong_arity.esk" \
      -o "${e1_tmp}/wrong-arity-${run}.o" \
      >"${e1_tmp}/wrong-arity-${run}.log" 2>&1; then
    die "E1 wrong-arity accessor unexpectedly compiled"
  fi
  grep -F 'Arity mismatch: transformer-error-category expects 1 arguments but got 0' \
    "${e1_tmp}/wrong-arity-${run}.log" >/dev/null
  test ! -e "${e1_tmp}/wrong-arity-${run}.o"
done
cmp "${e1_tmp}/public-leak-1.log" "${e1_tmp}/public-leak-2.log"
cmp "${e1_tmp}/capabilities-leak-1.log" "${e1_tmp}/capabilities-leak-2.log"
cmp "${e1_tmp}/wrong-arity-1.log" "${e1_tmp}/wrong-arity-2.log"

printf 'E1 PASS: 106 runtime checks; deterministic AOT/JIT; declared public boundary verified\n'
