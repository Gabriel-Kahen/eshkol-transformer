#!/usr/bin/env bash
# Sourced by the terminal runner. Each invocation gets immutable raw streams.
terminal_record_command() {
  local label=$1 stdout_alias=$2 stderr_alias=$3 receipt status
  shift 3
  [[ ${1:-} == -- ]] || return 2
  shift
  [[ $# -gt 0 && $label =~ ^[a-zA-Z0-9_-]+$ ]] || return 2
  terminal_command_index=$((terminal_command_index + 1))
  printf -v receipt '%s/%04d-%s' \
    "$terminal_commands_root" "$terminal_command_index" "$label"
  mkdir -- "$receipt" || return
  printf '%s\n' "$label" >"$receipt/label"
  pwd -P >"$receipt/cwd"
  printf '%s\0' "$@" >"$receipt/argv.nul"
  if "$@" >"$receipt/stdout.raw" 2>"$receipt/stderr.raw"; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" >"$receipt/exit"
  if [[ $stdout_alias != - ]]; then
    ln -f -- "$receipt/stdout.raw" "$evidence/$stdout_alias" || return
  fi
  if [[ $stderr_alias != - ]]; then
    ln -f -- "$receipt/stderr.raw" "$evidence/$stderr_alias" || return
  fi
  return "$status"
}

# verify_toolchain must run in a child to preserve a receipt even when die()
# exits. Restore the only caller-side mutation made by common.sh's function.
terminal_restore_toolchain_export() {
  export ESHKOL_CXX_COMPILER
  ESHKOL_CXX_COMPILER="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
}

# Eshkol invokes the C++ linker through a generated wrapper in another
# process. Its atomically named receipt cannot race the parent's index.
terminal_record_nested_link() {
  local prefix=$1 receipt status replay_status=0
  shift
  [[ $# -gt 0 && $prefix =~ ^[a-zA-Z0-9_-]+$ ]] || return 2
  receipt=$(mktemp -d "$terminal_commands_root/$prefix-clang-link.XXXXXX") || return
  printf '%s\n' "$prefix-clang-link" >"$receipt/label"
  pwd -P >"$receipt/cwd"
  printf '%s\0' "$@" >"$receipt/argv.nul"
  if "$@" >"$receipt/stdout.raw" 2>"$receipt/stderr.raw"; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" >"$receipt/exit"
  cat -- "$receipt/stdout.raw" || replay_status=$?
  cat -- "$receipt/stderr.raw" >&2 || replay_status=$?
  if (( status != 0 )); then return "$status"; fi
  return "$replay_status"
}
