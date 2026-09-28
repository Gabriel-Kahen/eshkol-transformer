#!/usr/bin/env bash
# E1B-only fixed G3-G G0 public tuple. Partial or mixed package inputs reject.
g3g_g0_prefix="${PROJECT_ROOT}/native/g3g_g0_package"
g3g_g0_inputs=("${g3g_g0_prefix}_root.esk" "${g3g_g0_prefix}_bridge.c"
            "${g3g_g0_prefix}_private_renames.txt" "${g3g_g0_prefix}_public_exports.txt")
g3g_g0_tuple_requested=0
for g3g_g0_raw in "${raw_private_root}" "${raw_package_bridge}" \
    "${raw_package_renames}" "${raw_public_exports}"; do
  for g3g_g0_expected in "${g3g_g0_inputs[@]}"; do
    [[ "$(realpath -m -- "${g3g_g0_raw}")" != "${g3g_g0_expected}" ]] || \
      g3g_g0_tuple_requested=1
  done
done
if [[ "${g3g_g0_tuple_requested}" == 1 ]]; then
  [[ "${raw_private_root}" == "${g3g_g0_inputs[0]}" && \
     "${raw_package_bridge}" == "${g3g_g0_inputs[1]}" && \
     "${raw_package_renames}" == "${g3g_g0_inputs[2]}" && \
     "${raw_public_exports}" == "${g3g_g0_inputs[3]}" ]] || \
    die "G3-G G0 requires the exact repository tuple"
  [[ "${#raw_include_dirs[@]}" == 4 && \
     "${raw_include_dirs[0]}" == "${PROJECT_ROOT}/internal/p1/lib" && \
     "${raw_include_dirs[1]}" == "${PROJECT_ROOT}/internal/c1/lib" && \
     "${raw_include_dirs[2]}" == "${PROJECT_ROOT}/internal/t1/lib" && \
     "${raw_include_dirs[3]}" == "${PROJECT_ROOT}/src" ]] || \
    die "G3-G G0 requires exact ordered trusted include roots"
  for g3g_g0_path in "${g3g_g0_inputs[@]}" "${raw_include_dirs[@]}" \
      "${g3g_g0_prefix}_source_closure.txt" \
      "${g3g_g0_prefix}_native_source_closure.txt" \
      "${g3g_g0_prefix}_undefined_symbols.txt" \
      "${g3g_g0_prefix}_public_strings.txt"; do
    m3_check_repository_path "${g3g_g0_path}"
  done
  for g3g_g0_manifest in "${g3g_g0_prefix}_source_closure.txt" \
      "${g3g_g0_prefix}_native_source_closure.txt"; do
    [[ -s "${g3g_g0_manifest}" ]] || die "G3-G G0 reviewed closure missing"
    while IFS= read -r g3g_g0_relative; do
      [[ -n "${g3g_g0_relative}" && "${g3g_g0_relative}" != /* && \
         "${g3g_g0_relative}" != *'..'* ]] || die "G3-G G0 malformed closure entry"
      m3_check_repository_path "${PROJECT_ROOT}/${g3g_g0_relative}"
    done <"${g3g_g0_manifest}"
  done
fi
