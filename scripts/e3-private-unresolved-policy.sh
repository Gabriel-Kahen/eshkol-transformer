#!/usr/bin/env bash
# E1B has already authenticated the exact E3 tuple and compared its complete
# undefined inventory. The retention/overflow region observer is supplied by the
# separately compiled, noninstalled C++ test object at final link.
e3_reject_unresolved_private() {
  local tuple_kind=$1 actual=$2 expected=$3 symbol
  while IFS= read -r symbol; do
    [[ "${symbol}" =~ ^et_(e3|d2|l2|l3s|m3|m3t|n2|n3k|a2|f32|i64|p1|kernel)_ ]] ||
      continue
    if [[ ( "${tuple_kind}" == retention ||
            "${tuple_kind}" == r1-d2-batch-retry ||
            "${tuple_kind}" == retention-first-eval-mutant ||
            "${tuple_kind}" == retention-overflow ) &&
          "${symbol}" == et_e3_retention_region_used_v1 ]] &&
       grep -Fxq -- "${symbol}" "${expected}"; then
      continue
    fi
    die "E3 aggregate retains unresolved private native authority: ${symbol}"
  done <"${actual}"
}
