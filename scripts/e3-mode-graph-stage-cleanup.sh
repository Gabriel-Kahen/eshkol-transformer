#!/usr/bin/env bash
# Sourced only by the noninstalled mode/graph and retention package builders.
e3_retention_generated_stage() {
  local object=$1 stage gate generated
  stage="$(dirname -- "${object}")"
  gate="$(dirname -- "${stage}")"
  [[ "$(basename -- "${object}")" == e3_retention_package.o &&
     "$(basename -- "${stage}")" =~ ^\.e3-retention\.[[:alnum:]]{6}$ &&
     -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     -d "${gate}" && ! -L "${gate}" && -O "${gate}" ]] || return 1
  generated="${gate}/.e3-retention-generated"
  [[ ! -e "${generated}" && ! -L "${generated}" ]] || return 1
  printf '%s\n' "${generated}"
}

e3_mode_graph_stage_cleanup() {
  local status=$?
  trap - EXIT
  if ((status == 0)); then
    rm -rf -- "${e3_mode_graph_stage_dir}"
  else
    if [[ -d "${e3_mode_graph_stage_dir}" ]]; then
      printf '%d\n' "${status}" >"${e3_mode_graph_stage_dir}/exit.status" || true
    fi
    printf 'E3 %s failed staging retained at %s (status %d)\n' \
      "${e3_mode_graph_stage_label}" "${e3_mode_graph_stage_dir}" "${status}" >&2
  fi
  exit "${status}"
}

e3_mode_graph_install_stage_cleanup() {
  e3_mode_graph_stage_dir=${1:?staging directory required}
  e3_mode_graph_stage_label=${2:-mode/graph}
  trap e3_mode_graph_stage_cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
}
