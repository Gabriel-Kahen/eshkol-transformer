#!/usr/bin/env bash
# The noninstalled uniform driver alone adds these four compiled imports.
e3_uniform_verify_import_manifests() {
  [[ "${e3_tuple_kind:-}" == uniform ]] ||
    die "E3 uniform import policy rejects unrelated tuple"
  local base=$1 uniform=$2 base_added=$3 uniform_added=$4
  local base_absent=$5 uniform_absent=$6 manifest
  for manifest in "${base}" "${uniform}" "${base_added}" "${uniform_added}" \
      "${base_absent}" "${uniform_absent}"; do
    [[ -s "${manifest}" ]] || die "E3 uniform import manifest is empty"
    LC_ALL=C sort -u "${manifest}" | cmp -s "${manifest}" - ||
      die "E3 uniform import manifest is not sorted unique text"
  done
  [[ "$(wc -l <"${uniform}")" == 177 ]] ||
    die "E3 uniform normal import count differs"
  [[ ! -s <(comm -23 "${base}" "${uniform}") ]] ||
    die "E3 uniform removed a numerical import"
  comm -13 "${base}" "${uniform}" |
    cmp -s - <(printf '%s\n' eshkol_display_value \
      eshkol_runtime_current_output_fp eshkol_tail_transfer_slot fputc) ||
    die "E3 uniform driver import delta differs"
  cmp -s "${base_added}" "${uniform_added}" ||
    die "E3 uniform sanitizer added-import delta differs"
  cmp -s "${base_absent}" "${uniform_absent}" ||
    die "E3 uniform sanitizer absent-import delta differs"
}
