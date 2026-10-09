#!/usr/bin/env bash
# A fresh, noninstalled sealed-fixture gate owns the stable generated-source path.
e3_sealed_fixture_generated_stage() {
  local object=$1 stage gate generated
  stage="$(dirname -- "${object}")"
  gate="$(dirname -- "${stage}")"
  [[ "$(basename -- "${object}")" == e3_sealed_fixture_package.o &&
     "$(basename -- "${stage}")" =~ ^\.e3-sealed-fixture\.[[:alnum:]]{6}$ &&
     -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     -d "${gate}" && ! -L "${gate}" && -O "${gate}" ]] || return 1
  generated="${gate}/.e3-sealed-fixture-generated"
  [[ ! -e "${generated}" && ! -L "${generated}" ]] || return 1
  printf '%s\n' "${generated}"
}
