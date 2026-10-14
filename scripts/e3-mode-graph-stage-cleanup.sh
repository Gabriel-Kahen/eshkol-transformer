#!/usr/bin/env bash
# Private mode/graph and retention build staging and cleanup helpers.
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

e3_fault_generated_stage() {
  local object=$1 stage gate generated
  stage="$(dirname -- "${object}")"
  gate="$(dirname -- "${stage}")"
  [[ "$(basename -- "${object}")" == e3_fault_package.o &&
     "$(basename -- "${stage}")" =~ ^\.e3-fault\.[[:alnum:]]{6}$ &&
     -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     -d "${gate}" && ! -L "${gate}" && -O "${gate}" ]] || return 1
  generated="${gate}/.e3-fault-generated"
  [[ ! -e "${generated}" && ! -L "${generated}" ]] || return 1
  printf '%s\n' "${generated}"
}

e3_overflow_generated_stage() {
  local object=$1 stage gate generated
  stage="$(dirname -- "${object}")"
  gate="$(dirname -- "${stage}")"
  [[ "$(basename -- "${object}")" == e3_overflow_package.o &&
     "$(basename -- "${stage}")" =~ ^\.e3-overflow\.[[:alnum:]]{6}$ &&
     -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     -d "${gate}" && ! -L "${gate}" && -O "${gate}" ]] || return 1
  generated="${gate}/.e3-overflow-generated"
  [[ ! -e "${generated}" && ! -L "${generated}" ]] || return 1
  printf '%s\n' "${generated}"
}

e3_mode_graph_generated_stage() {
  local object=$1 stage gate generated
  stage="$(dirname -- "${object}")"
  gate="$(dirname -- "${stage}")"
  [[ "$(basename -- "${object}")" == e3_mode_graph_package.o &&
     "$(basename -- "${stage}")" =~ ^\.e3-mode-graph\.[[:alnum:]]{6}$ &&
     -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     -d "${gate}" && ! -L "${gate}" && -O "${gate}" ]] || return 1
  generated="${gate}/.e3-mode-graph-generated"
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
