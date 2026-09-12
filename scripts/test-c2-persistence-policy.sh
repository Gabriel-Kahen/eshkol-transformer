#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in cmp env rg timeout; do require_command "${command}"; done

runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-policy.XXXXXX")"
trap 'rm -rf -- "${tmp}"' EXIT

compile_policy() {
  local label=$1
  mkdir -p "${tmp}/${label}"
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${tmp}/cache-${label}" \
    timeout --foreground --signal=TERM --kill-after=5s 180s \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
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
  rg -x 'C2 PERSISTENCE POLICY PASS: 71 checks' \
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

if rg -n '^\(provide|^\(extern|et_e1b_public_|lib/transformer/persistence\.esk' \
    "${PROJECT_ROOT}/native/c2_persistence_policy_extension.esk" \
    "${PROJECT_ROOT}/native/c2_persistence_policy_source_closure.txt"; then
  die "private C2 policy closure publishes a facade or native ABI"
fi

printf 'C2 PERSISTENCE POLICY PASS: strict repeated source-composed AOT/runtime, exact identity, hard/operational boundaries, C1 projections, source closure\n'
