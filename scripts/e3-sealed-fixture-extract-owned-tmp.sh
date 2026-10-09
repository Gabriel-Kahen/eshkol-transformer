#!/usr/bin/env bash
# Keep both successful emissions and failed builder scratch on the caller's
# persistent stage. A failed raw write must never yield apparent success.
e3_extract_owned_command() (
  local stage=$1 result stdout_id stderr_id output_fd error_fd helper_dir
  shift
  [[ -d "${stage}" && ! -L "${stage}" && -O "${stage}" &&
     ! -e "${stage}/tmp" && ! -L "${stage}/tmp" ]] || return 1
  mkdir -m 0700 -- "${stage}/tmp" || return 1
  helper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
  set -C
  { printf '%q ' "$@" || return 1; printf '\n' || return 1; } \
    >"${stage}/compiler.argv.raw" || return 1
  exec {output_fd}>"${stage}/compiler.stdout.raw" || return 1
  exec {error_fd}>"${stage}/compiler.stderr.raw" || return 1
  stdout_id="$(stat -c '%d:%i' -- "${stage}/compiler.stdout.raw")" || return 1
  stderr_id="$(stat -c '%d:%i' -- "${stage}/compiler.stderr.raw")" || return 1
  set +C
  if TMPDIR="${stage}/tmp" "$@" >&"${output_fd}" 2>&"${error_fd}"; then
    result=0
  else
    result=$?
  fi
  exec {output_fd}>&- || return 1
  exec {error_fd}>&- || return 1
  /usr/bin/python3 -B "${helper_dir}/e3-sealed-fixture-check-receipts.py" \
    --write-exit "${stage}" "${result}" || return 1
  [[ "$(stat -c '%d:%i' -- "${stage}/compiler.stdout.raw")" == "${stdout_id}" &&
     "$(stat -c '%d:%i' -- "${stage}/compiler.stderr.raw")" == "${stderr_id}" ]] || return 1
  /usr/bin/python3 -B "${helper_dir}/e3-sealed-fixture-check-receipts.py" \
    "${stage}/compiler.argv.raw" "${stage}/compiler.stdout.raw" \
    "${stage}/compiler.stderr.raw" "${stage}/compiler.exit.raw" || return 1
  return "${result}"
)
