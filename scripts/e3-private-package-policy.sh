#!/usr/bin/env bash
# Sourced only by E1B. This admits the base private E3 tuple or its exact
# source-private diagnostic successor. Mixed and partial tuples are rejected.
e3_base_prefix="${PROJECT_ROOT}/native/e3_private_package"
e3_diagnostic_prefix="${PROJECT_ROOT}/native/e3_diagnostic_private_package"
e3_base_inputs=(
  "${PROJECT_ROOT}/native/e3_private_driver_root.esk"
  "${PROJECT_ROOT}/native/e3_private_bridge.c"
  "${e3_base_prefix}_private_renames.txt"
  "${e3_base_prefix}_public_exports.txt"
)
e3_diagnostic_inputs=(
  "${PROJECT_ROOT}/native/e3_diagnostic_private_driver_root.esk"
  "${PROJECT_ROOT}/native/e3_diagnostic_private_bridge.c"
  "${e3_diagnostic_prefix}_private_renames.txt"
  "${e3_diagnostic_prefix}_public_exports.txt"
)
e3_prefix="${e3_base_prefix}"
e3_inputs=("${e3_base_inputs[@]}")
e3_tuple_requested=0
e3_tuple_kind=
e3_diagnostic_tuple=0
for e3_raw in "${raw_private_root}" "${raw_package_bridge}" \
    "${raw_package_renames}" "${raw_public_exports}"; do
  for e3_expected in "${e3_base_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == base ]] || \
        die "E3 policy rejects mixed base and diagnostic tuples"
      e3_tuple_kind=base
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_diagnostic_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == diagnostic ]] || \
        die "E3 policy rejects mixed base and diagnostic tuples"
      e3_tuple_kind=diagnostic
      e3_tuple_requested=1
    fi
  done
done

if [[ "${e3_tuple_kind}" == diagnostic ]]; then
  e3_prefix="${e3_diagnostic_prefix}"
  e3_inputs=("${e3_diagnostic_inputs[@]}")
  e3_diagnostic_tuple=1
fi

e3_check_repository_path() {
  local path=$1 part current="${PROJECT_ROOT}"
  [[ "${path}" == "${PROJECT_ROOT}/"* && "${path}" != *'/../'* && \
     "${path}" != *'/./'* && "$(realpath -m -- "${path}")" == "${path}" ]] || \
    die "E3 policy requires exact canonical repository paths"
  local relative=${path#"${PROJECT_ROOT}/"}
  local -a parts
  IFS=/ read -r -a parts <<<"${relative}"
  for part in "${parts[@]}"; do
    current+="/${part}"
    [[ ! -L "${current}" ]] || die "E3 policy rejects symlinked repository inputs"
  done
}

e3_check_native_dependency_path() {
  local raw=$1 current=/ part
  local -a parts
  [[ "${raw}" == /* ]] || die "E3 native depfile path must be absolute"
  IFS=/ read -r -a parts <<<"${raw}"
  for part in "${parts[@]}"; do
    case "${part}" in
      ''|.) continue ;;
      ..) current="${current%/*}"; [[ -n "${current}" ]] || current=/ ;;
      *) current="${current%/}/${part}"
         [[ ! -L "${current}" ]] || die "E3 native depfile uses a symlink alias" ;;
    esac
  done
}

e3_normalize_source_dependencies() {
  local dependency generated
  [[ -n "${e3_generated_source_root:-}" ]] || \
    die "E3 private generated source was not initialized"
  generated="${e3_generated_source_root}/e3_d2_dataset.esk"
  while IFS= read -r dependency; do
    [[ -n "${dependency}" ]] || continue
    if [[ "$(realpath -m -- "${dependency}")" == "${generated}" ]]; then
      printf '@BUILD@/source/e3_d2_dataset.esk\n'
    else
      e3_check_repository_path "${dependency}"
      printf '%s\n' "${dependency#"${PROJECT_ROOT}/"}"
    fi
  done
}

if [[ "${e3_tuple_requested}" == 1 ]]; then
  [[ "${raw_private_root}" == "${e3_inputs[0]}" && \
     "${raw_package_bridge}" == "${e3_inputs[1]}" && \
     "${raw_package_renames}" == "${e3_inputs[2]}" && \
     "${raw_public_exports}" == "${e3_inputs[3]}" ]] || \
    die "E3 policy requires the exact lexical repository tuple"
  [[ "${#raw_include_dirs[@]}" == 6 && \
     "${raw_include_dirs[0]}" == "${PROJECT_ROOT}/internal/p1/lib" && \
     "${raw_include_dirs[1]}" == "${PROJECT_ROOT}/internal/c1/lib" && \
     "${raw_include_dirs[2]}" == "${PROJECT_ROOT}/internal/t1/lib" && \
     "${raw_include_dirs[3]}" == "${PROJECT_ROOT}/src" && \
     "${raw_include_dirs[4]}" == "${PROJECT_ROOT}/internal/d2/lib" && \
     "${raw_include_dirs[5]}" == "${PROJECT_ROOT}/internal/e3/lib" ]] || \
    die "E3 policy requires exact ordered trusted include roots"
  for e3_path in "${e3_inputs[@]}" "${raw_include_dirs[@]}" \
      "${e3_prefix}_defined_symbols.txt" \
      "${e3_prefix}_undefined_symbols.txt" \
      "${e3_prefix}_public_strings.txt" \
      "${e3_prefix}_native_objects.txt" \
      "${e3_prefix}_archive_members.txt" \
      "${e3_prefix}_source_closure.txt" \
      "${e3_prefix}_native_source_closure.txt"; do
    e3_check_repository_path "${e3_path}"
  done
fi
