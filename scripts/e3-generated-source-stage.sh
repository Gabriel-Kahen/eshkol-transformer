#!/usr/bin/env bash
# Select a stable generated-source path only for a private, gate-owned stage.
e3_numerical_generated_stage() {
  local object=$1 stage gate generated
  stage="$(dirname -- "${object}")"
  gate="$(dirname -- "${stage}")"
  [[ "$(basename -- "${object}")" == e3_numerical_package.o &&
     "$(basename -- "${stage}")" =~ ^\.e3-numerical\.[[:alnum:]]{6}$ &&
     -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     -d "${gate}" && ! -L "${gate}" && -O "${gate}" ]] ||
    return 1
  generated="${gate}/.e3-numerical-generated"
  [[ ! -e "${generated}" && ! -L "${generated}" ]] || return 1
  printf '%s\n' "${generated}"
}
