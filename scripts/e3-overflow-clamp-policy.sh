#!/usr/bin/env bash
# Sourced by E1B after exact tuple admission. No caller-supplied source path.
e3_overflow_mutant=normal
if [[ -v E3_OVERFLOW_MUTANT ]]; then
  [[ "${E3_OVERFLOW_MUTANT}" == clamp &&
     "${package_policy}" == e3-private-aggregate &&
     "${e3_tuple_kind}" == retention-overflow &&
     "${e3_private_test_sanitize}" == 0 ]] ||
    die "E3 clamp mutation requires the unsanitized exact overflow tuple"
  e3_overflow_mutant=clamp
fi

e3_overflow_clamp_dependency() {
  local path=$1
  [[ "${e3_overflow_mutant}" == clamp &&
     "${path}" == "${e3_overflow_clamp_source:-}" &&
     -f "${path}" && ! -L "${path}" ]] || return 1
  printf '%s/@BUILD@/native/e3_evaluation_metrics_provider.c\n' "${PROJECT_ROOT}"
}
