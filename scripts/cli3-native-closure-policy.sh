#!/usr/bin/env bash

# Resolve compiler-emitted C dependency spellings without admitting an alias or
# a traversal outside the trusted repository tree.
cli3_canonical_native_dependency_path() {
  local raw=$1 current="${PROJECT_ROOT}" resolved part depth=0
  local -a parts
  [[ "${raw}" == "${PROJECT_ROOT}/"* ]] || \
    die "CLI3 native depfile path is outside the repository"
  IFS=/ read -r -a parts <<<"${raw#"${PROJECT_ROOT}/"}"
  for part in "${parts[@]}"; do
    case "${part}" in
      ''|.) continue ;;
      ..)
        (( depth > 0 )) || die "CLI3 native depfile path escapes the repository"
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
  [[ "${resolved}" == "${PROJECT_ROOT}/"* ]] || \
    die "CLI3 native depfile resolves outside the repository"
  printf '%s\n' "${resolved}"
}
