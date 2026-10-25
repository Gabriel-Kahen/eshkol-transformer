#!/usr/bin/env bash
# Exact measured imports for the noninstalled N5 test tuple.
e3_n5_verify_sanitizer_imports() {
  [[ "${e3_tuple_kind:-}" == n5 ]] ||
    die "E3 N5 import policy rejects unrelated tuple"
  local normal=$1 noninstrumentation=$2 instrumentation=$3
  local expected_non=$4 expected_instrumentation=$5 manifest
  for manifest in "${normal}" "${noninstrumentation}" "${instrumentation}" \
      "${expected_non}" "${expected_instrumentation}"; do
    [[ -s "${manifest}" ]] || die "E3 N5 import manifest is empty"
    LC_ALL=C sort -u "${manifest}" | cmp -s "${manifest}" - ||
      die "E3 N5 import manifest is not sorted unique text"
  done
  [[ "$(wc -l <"${normal}")" == 172 &&
     "$(wc -l <"${expected_non}")" == 166 &&
     "$(wc -l <"${expected_instrumentation}")" == 57 ]] ||
    die "E3 N5 import counts differ from discovery"
  comm -13 "${normal}" "${expected_non}" |
    cmp -s - <(printf '%s\n' bcmp puts) ||
    die "E3 N5 noninstrumentation added-import delta differs"
  comm -23 "${normal}" "${expected_non}" |
    cmp -s - <(printf '%s\n' eshkol_ad_seed_flag eshkol_runtime_fatal \
      eshkol_taylor_lift_ad_node memcmp memcpy memmove memset printf) ||
    die "E3 N5 noninstrumentation absent-import delta differs"
  cmp -s "${expected_non}" "${noninstrumentation}" ||
    die "E3 N5 sanitized noninstrumentation imports differ"
  cmp -s "${expected_instrumentation}" "${instrumentation}" ||
    die "E3 N5 sanitizer instrumentation imports differ"
}
