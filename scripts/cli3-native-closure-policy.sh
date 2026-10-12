#!/usr/bin/env bash

# Resolve compiler-emitted C dependency spellings without admitting an alias or
# a traversal outside either authenticated source root.
cli3_canonical_native_dependency_path() {
  local raw=$1 root current resolved part depth=0
  local -a parts
  case "${raw}" in
    "${PROJECT_ROOT}/"*) root=${PROJECT_ROOT} ;;
    "${e1b_source}/"*) root=${e1b_source} ;;
    *) die "CLI3 native depfile path is outside reviewed source roots" ;;
  esac
  current=${root}
  IFS=/ read -r -a parts <<<"${raw#"${root}/"}"
  for part in "${parts[@]}"; do
    case "${part}" in
      ''|.) continue ;;
      ..)
        (( depth > 0 )) || die "CLI3 native depfile path escapes its reviewed source root"
        depth=$((depth - 1))
        current="${current%/*}"
        ;;
      *)
        current="${current}/${part}"
        [[ ! -L "${current}" ]] || die "CLI3 native depfile uses a symlink alias"
        depth=$((depth + 1))
        ;;
    esac
  done
  resolved="$(realpath -- "${raw}")" || die "CLI3 native depfile path is unavailable"
  [[ "${resolved}" == "${root}/"* ]] || \
    die "CLI3 native depfile resolves outside its reviewed source root"
  printf '%s\n' "${resolved}"
}
