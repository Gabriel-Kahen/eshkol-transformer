#!/usr/bin/env bash
# Only an exact fresh zero-control gate staging child owns this stable path.
e3_zero_generated_stage() {
  local object=$1 stage gate generated
  stage="$(dirname -- "${object}")"
  gate="$(dirname -- "${stage}")"
  [[ "$(basename -- "${object}")" == e3_zero_control_package.o &&
     "$(basename -- "${stage}")" =~ ^\.e3-zero-control\.[[:alnum:]]{6}$ &&
     -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     -d "${gate}" && ! -L "${gate}" && -O "${gate}" ]] || return 1
  generated="${gate}/.e3-zero-control-generated"
  [[ ! -e "${generated}" && ! -L "${generated}" ]] || return 1
  printf '%s\n' "${generated}"
}
