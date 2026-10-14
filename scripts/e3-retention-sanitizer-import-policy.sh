#!/usr/bin/env bash
# Only the exact noninstalled retention test tuples may use these inventories.
# The retention sanitizer split originated from the 27d9a98 Clang 21 object.
# The later edge phase diagnostics added four ordinary port/write imports to
# each declared tuple; exact compiled inventory still requires a fresh gate.
e3_retention_verify_sanitizer_imports() {
  local normal=$1 noninstrumentation=$2 instrumentation=$3
  local prefix non_count instrumentation_count absent_count
  case "${e3_tuple_kind:-retention}" in
    retention)
      prefix="${PROJECT_ROOT}/tests/e3_private/retention_package_sanitized"
      non_count=176 instrumentation_count=59 absent_count=8 ;;
    retention-first-eval-mutant)
      prefix="${PROJECT_ROOT}/tests/e3_private/retention_first_eval_mutant_package_sanitized"
      non_count=179 instrumentation_count=51 absent_count=5 ;;
    *) die "E3 retention sanitizer policy rejects unrelated tuple" ;;
  esac
  local expected_non="${prefix}_noninstrumentation_undefined_symbols.txt"
  local expected_instrumentation="${prefix}_instrumentation_undefined_symbols.txt"
  local manifest
  for manifest in "${expected_non}" "${expected_instrumentation}"; do
    [[ -s "${manifest}" ]] || die "E3 retention sanitizer import manifest is empty"
    LC_ALL=C sort -u "${manifest}" | cmp -s "${manifest}" - ||
      die "E3 retention sanitizer import manifest is not sorted unique text"
  done
  [[ "$(awk 'END {print NR}' "${expected_non}")" == "${non_count}" &&
     "$(awk 'END {print NR}' "${expected_instrumentation}")" == "${instrumentation_count}" ]] ||
    die "E3 retention sanitizer manifest counts differ from the declared tuple"
  cmp -s "${expected_non}" "${noninstrumentation}" ||
    die "E3 retention sanitized noninstrumentation imports differ"
  cmp -s "${expected_instrumentation}" "${instrumentation}" ||
    die "E3 retention sanitizer instrumentation imports differ"
  comm -13 "${normal}" "${noninstrumentation}" |
    awk 'END {if (NR != 2) exit 1}' ||
    die "E3 retention sanitizer added import count differs"
  comm -23 "${normal}" "${noninstrumentation}" |
    awk -v expected="${absent_count}" 'END {if (NR != expected) exit 1}' ||
    die "E3 retention sanitizer absent import count differs"
}
