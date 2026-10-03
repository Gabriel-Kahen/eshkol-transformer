#!/usr/bin/env bash
# Sourced only by E1B's exact, noninstalled E3 mode/graph sanitizer tuple.
e3_mode_graph_verify_sanitizer_imports() {
  local normal=$1 noninstrumentation=$2 instrumentation=$3
  local expected_normal="${PROJECT_ROOT}/tests/e3_mode_graph/package_undefined_symbols.txt"
  local prefix="${PROJECT_ROOT}/tests/e3_mode_graph/package_sanitized"
  local expected_non="${prefix}_noninstrumentation_undefined_symbols.txt"
  local expected_instrumentation="${prefix}_instrumentation_undefined_symbols.txt"
  local manifest
  for manifest in "${expected_normal}" "${expected_non}" "${expected_instrumentation}"; do
    [[ -s "${manifest}" ]] || die "E3 mode/graph sanitizer import manifest is empty"
    LC_ALL=C sort -u "${manifest}" | cmp -s "${manifest}" - ||
      die "E3 mode/graph sanitizer import manifest is not sorted unique text"
  done
  [[ "$(awk 'END {print NR}' "${expected_normal}")" == 175 &&
     "$(awk 'END {print NR}' "${expected_non}")" == 172 &&
     "$(awk 'END {print NR}' "${expected_instrumentation}")" == 51 ]] ||
    die "E3 mode/graph sanitizer manifest counts differ from the sealed tuple"
  cmp -s "${expected_normal}" "${normal}" ||
    die "E3 mode/graph normal imports differ from the sealed tuple"
  cmp -s "${expected_non}" "${noninstrumentation}" ||
    die "E3 mode/graph sanitized noninstrumentation imports differ"
  cmp -s "${expected_instrumentation}" "${instrumentation}" ||
    die "E3 mode/graph sanitizer instrumentation imports differ"
  comm -13 "${normal}" "${noninstrumentation}" |
    awk 'END {if (NR != 2) exit 1}' ||
    die "E3 mode/graph sanitizer added import count differs"
  comm -23 "${normal}" "${noninstrumentation}" |
    awk 'END {if (NR != 5) exit 1}' ||
    die "E3 mode/graph sanitizer absent import count differs"
}
