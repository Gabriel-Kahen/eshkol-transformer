#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar cc cmp env rg timeout; do require_command "${command}"; done

runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-policy.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,240p' {} \; >&2 || true
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT
runtime="${tmp}/runtime"
mkdir -p "${runtime}"
for source in kernel_abi f32_tensor; do
  cc -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
    -fvisibility=hidden -fno-common -I "${PROJECT_ROOT}/include" \
    -I "${PROJECT_ROOT}/native" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
cc -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -fvisibility=hidden -fno-common -DET_C2_CARRIER_FACTORIES \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/k2_capabilities.c" \
  -o "${runtime}/k2_capabilities.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_policy_auth.a" \
  "${runtime}"/*.o

compile_policy() {
  local label=$1
  mkdir -p "${tmp}/${label}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${tmp}/cache-${label}" \
    timeout --foreground --signal=TERM --kill-after=5s 180s \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
      -L "${runtime}" --lib eshkol_transformer_c2_policy_auth \
      "${PROJECT_ROOT}/tests/c2/c2_persistence_policy_runtime.esk" \
      -o "${tmp}/${label}/policy" \
      >"${tmp}/${label}/compile.stdout" \
      2>"${tmp}/${label}/compile.stderr"
  test ! -s "${tmp}/${label}/compile.stderr"
}

compile_policy a
compile_policy b
cmp "${tmp}/a/policy" "${tmp}/b/policy"

for label in a b; do
  ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 60s \
      "${tmp}/${label}/policy" \
      >"${tmp}/${label}/run.stdout" 2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  rg -x 'C2 PERSISTENCE POLICY PASS: 73 checks' \
    "${tmp}/${label}/run.stdout" >/dev/null
done
cmp "${tmp}/a/run.stdout" "${tmp}/b/run.stdout"

cmp "${PROJECT_ROOT}/native/c2_persistence_policy_source_closure.txt" \
  <(printf '%s\n' \
    native/e1b_error_consumer_private.esk \
    lib/transformer/error_internal.esk \
    lib/transformer/error_core.esk \
    internal/c1/lib/transformer/persistence_policy_internal.esk \
    native/c2_persistence_policy_extension.esk)

if rg -n '^\(provide|et_e1b_public_|lib/transformer/persistence\.esk' \
    "${PROJECT_ROOT}/native/c2_persistence_policy_extension.esk" \
    "${PROJECT_ROOT}/native/c2_persistence_policy_source_closure.txt"; then
  die "private C2 policy closure publishes a facade"
fi
test "$(rg -c ':real et_c2_private_policy_factory_(register|authenticate)_v1' \
  "${PROJECT_ROOT}/native/c2_persistence_policy_extension.esk")" -eq 2

printf 'C2 PERSISTENCE POLICY PASS: strict repeated source-composed AOT/runtime, exact identity, hard/operational boundaries, C1 projections, source closure\n'
