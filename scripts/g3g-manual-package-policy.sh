#!/usr/bin/env bash
# E1B-only fixed G3-G manual public tuple. Partial or mixed package inputs reject.
g3g_manual_prefix="${PROJECT_ROOT}/native/g3g_manual_package"
g3g_manual_inputs=("${g3g_manual_prefix}_root.esk" "${g3g_manual_prefix}_bridge.c"
            "${g3g_manual_prefix}_private_renames.txt" "${g3g_manual_prefix}_public_exports.txt")
g3g_manual_tuple_requested=0
for g3g_manual_raw in "${raw_private_root}" "${raw_package_bridge}" \
    "${raw_package_renames}" "${raw_public_exports}"; do
  for g3g_manual_expected in "${g3g_manual_inputs[@]}"; do
    [[ "$(realpath -m -- "${g3g_manual_raw}")" != "${g3g_manual_expected}" ]] || \
      g3g_manual_tuple_requested=1
  done
done
if [[ "${g3g_manual_tuple_requested}" == 1 ]]; then
  [[ "${raw_private_root}" == "${g3g_manual_inputs[0]}" && \
     "${raw_package_bridge}" == "${g3g_manual_inputs[1]}" && \
     "${raw_package_renames}" == "${g3g_manual_inputs[2]}" && \
     "${raw_public_exports}" == "${g3g_manual_inputs[3]}" ]] || \
    die "G3-G manual requires the exact repository tuple"
  [[ "${#raw_include_dirs[@]}" == 4 && \
     "${raw_include_dirs[0]}" == "${PROJECT_ROOT}/internal/p1/lib" && \
     "${raw_include_dirs[1]}" == "${PROJECT_ROOT}/internal/c1/lib" && \
     "${raw_include_dirs[2]}" == "${PROJECT_ROOT}/internal/t1/lib" && \
     "${raw_include_dirs[3]}" == "${PROJECT_ROOT}/src" ]] || \
    die "G3-G manual requires exact ordered trusted include roots"
  for g3g_manual_path in "${g3g_manual_inputs[@]}" "${raw_include_dirs[@]}" \
      "${g3g_manual_prefix}_source_closure.txt" \
      "${g3g_manual_prefix}_native_source_closure.txt" \
      "${g3g_manual_prefix}_undefined_symbols.txt" \
      "${g3g_manual_prefix}_public_strings.txt"; do
    m3_check_repository_path "${g3g_manual_path}"
  done
  for g3g_manual_manifest in "${g3g_manual_prefix}_source_closure.txt" \
      "${g3g_manual_prefix}_native_source_closure.txt"; do
    [[ -s "${g3g_manual_manifest}" ]] || die "G3-G manual reviewed closure missing"
    while IFS= read -r g3g_manual_relative; do
      [[ -n "${g3g_manual_relative}" && "${g3g_manual_relative}" != /* && \
         "${g3g_manual_relative}" != *'..'* ]] || die "G3-G manual malformed closure entry"
      m3_check_repository_path "${PROJECT_ROOT}/${g3g_manual_relative}"
    done <"${g3g_manual_manifest}"
  done
fi
