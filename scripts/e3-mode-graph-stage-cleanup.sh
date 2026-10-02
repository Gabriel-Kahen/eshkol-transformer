#!/usr/bin/env bash
# Sourced only by the noninstalled mode/graph package builder.
e3_mode_graph_stage_cleanup() {
  local status=$?
  trap - EXIT
  if ((status == 0)); then
    rm -rf -- "${e3_mode_graph_stage_dir}"
  else
    if [[ -d "${e3_mode_graph_stage_dir}" ]]; then
      printf '%d\n' "${status}" >"${e3_mode_graph_stage_dir}/exit.status" || true
    fi
    printf 'E3 mode/graph failed staging retained at %s (status %d)\n' \
      "${e3_mode_graph_stage_dir}" "${status}" >&2
  fi
  exit "${status}"
}

e3_mode_graph_install_stage_cleanup() {
  e3_mode_graph_stage_dir=${1:?staging directory required}
  trap e3_mode_graph_stage_cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
}
