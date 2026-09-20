#!/usr/bin/env bash
# Sourced only by E1B. This fixed repository tuple is not an object-input API.
m3_prefix="${PROJECT_ROOT}/native/m3_package"
m3_tuple_requested=0
m3_inputs=("${m3_prefix}_root.esk" "${m3_prefix}_bridge.c"
             "${m3_prefix}_private_renames.txt" "${m3_prefix}_public_exports.txt")
for m3_raw in "${raw_private_root}" "${raw_package_bridge}" \
    "${raw_package_renames}" "${raw_public_exports}"; do
  if [[ -f "${m3_raw}" ]] && \
      grep -Eq 'et_e1b_(public|private)_m3_' "${m3_raw}"; then
    m3_tuple_requested=1
  fi
  for m3_expected in "${m3_inputs[@]}"; do
    if [[ "$(realpath -m -- "${m3_raw}")" == "${m3_expected}" ]] || \
       { [[ -f "${m3_raw}" ]] && cmp -s "${m3_raw}" "${m3_expected}"; }; then
      m3_tuple_requested=1
    fi
  done
done
m3_check_repository_path() {
  local path=$1 part current="${PROJECT_ROOT}"
  [[ "${path}" == "${PROJECT_ROOT}/"* && "${path}" != *'/../'* && \
     "${path}" != *'/./'* && "$(realpath -m -- "${path}")" == "${path}" ]] || \
    die "M3 policy requires exact canonical repository paths"
  local relative=${path#"${PROJECT_ROOT}/"}
  local -a parts
  IFS=/ read -r -a parts <<<"${relative}"
  for part in "${parts[@]}"; do
    current+="/${part}"
    [[ ! -L "${current}" ]] || die "M3 policy rejects symlinked repository inputs"
  done
}
# Every emitted Eshkol dependency must belong to the exact repository closure.
# Do not filter first: that would erase evidence of an outside-root import.
m3_normalize_source_dependencies() {
  local dependency
  while IFS= read -r dependency; do
    [[ -n "${dependency}" ]] || continue
    m3_check_repository_path "${dependency}"
    printf '%s\n' "${dependency#"${PROJECT_ROOT}/"}"
  done
}
# Check the actual include spelling before realpath can erase a symlink alias.
# C includes may legitimately contain ../; inspect each component before popping it.
m3_check_native_dependency_path() {
  local raw=$1 current=/ part
  local -a parts
  [[ "${raw}" == /* ]] || die "M3 native depfile path must be absolute"
  IFS=/ read -r -a parts <<<"${raw}"
  for part in "${parts[@]}"; do
    case "${part}" in
      ''|.) continue ;;
      ..) current="${current%/*}"; [[ -n "${current}" ]] || current=/ ;;
      *)
        current="${current%/}/${part}"
        [[ ! -L "${current}" ]] || die "M3 native depfile uses a symlink alias"
        ;;
    esac
  done
}
if [[ "${m3_tuple_requested}" == 1 ]]; then
  [[ "${raw_private_root}" == "${m3_inputs[0]}" && \
     "${raw_package_bridge}" == "${m3_inputs[1]}" && \
     "${raw_package_renames}" == "${m3_inputs[2]}" && \
     "${raw_public_exports}" == "${m3_inputs[3]}" ]] || \
    die "M3 policy requires the exact lexical repository tuple"
  [[ "${#raw_include_dirs[@]}" == 4 && \
     "${raw_include_dirs[0]}" == "${PROJECT_ROOT}/internal/p1/lib" && \
     "${raw_include_dirs[1]}" == "${PROJECT_ROOT}/internal/c1/lib" && \
     "${raw_include_dirs[2]}" == "${PROJECT_ROOT}/internal/t1/lib" && \
     "${raw_include_dirs[3]}" == "${PROJECT_ROOT}/src" ]] || \
    die "M3 policy requires exact ordered trusted include roots"
  for m3_path in "${m3_inputs[@]}" "${raw_include_dirs[@]}" \
      "${m3_prefix}_defined_symbols.txt" "${m3_prefix}_undefined_symbols.txt" \
      "${m3_prefix}_public_strings.txt" "${m3_prefix}_native_objects.txt" \
      "${m3_prefix}_archive_members.txt" "${m3_prefix}_facades.txt"; do
    m3_check_repository_path "${m3_path}"
  done
  for m3_kind in source_closure native_source_closure; do
    m3_manifest="${m3_prefix}_${m3_kind}.txt"
    m3_check_repository_path "${m3_manifest}"
    [[ -s "${m3_manifest}" ]] || die "M3 reviewed closure manifest missing"
    while IFS= read -r m3_relative; do
      [[ -n "${m3_relative}" && "${m3_relative}" != /* && \
         "${m3_relative}" != *'..'* ]] || die "M3 malformed closure entry"
      m3_check_repository_path "${PROJECT_ROOT}/${m3_relative}"
    done <"${m3_manifest}"
  done
fi
