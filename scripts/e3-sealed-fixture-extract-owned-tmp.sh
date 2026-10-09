#!/usr/bin/env bash
# Keep both successful emissions and failed builder scratch on the caller's
# persistent, uniquely owned stage. No compiler command is selected here.
e3_extract_owned_command() {
  local stage=$1 result
  shift
  [[ -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     ! -e "${stage}/tmp" && ! -L "${stage}/tmp" ]] || return 1
  mkdir -m 0700 -- "${stage}/tmp" || return 1
  printf '%q ' "$@" >"${stage}/compiler.argv.raw"
  printf '\n' >>"${stage}/compiler.argv.raw"
  if TMPDIR="${stage}/tmp" "$@" >"${stage}/compiler.stdout.raw" \
      2>"${stage}/compiler.stderr.raw"; then
    printf '0\n' >"${stage}/compiler.exit.raw"
    return 0
  else
    result=$?
    printf '%s\n' "${result}" >"${stage}/compiler.exit.raw"
    return "${result}"
  fi
}
