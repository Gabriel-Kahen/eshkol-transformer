#!/usr/bin/bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

readonly -a c2_groups=(
  c2-format c2-state c2-save c2-load c2-public c2-operational
)

usage() {
  printf 'usage: %s [--group GROUP]...\n' "$0" >&2
  printf 'groups: %s\n' "${c2_groups[*]}" >&2
}

declare -A selected=()
if (( $# == 0 )); then
  for group in "${c2_groups[@]}"; do
    selected["${group}"]=1
  done
else
  while (( $# > 0 )); do
    if [[ "$1" != --group || $# -lt 2 ]]; then
      usage
      exit 2
    fi
    group=$2
    shift 2
    case "${group}" in
      c2-format|c2-state|c2-save|c2-load|c2-public|c2-operational) ;;
      *)
        usage
        exit 2
        ;;
    esac
    if [[ -n "${selected[${group}]:-}" ]]; then
      printf 'error: duplicate C2 group: %s\n' "${group}" >&2
      exit 2
    fi
    selected["${group}"]=1
  done
fi

run_gate() {
  /usr/bin/bash "${PROJECT_ROOT}/scripts/$1" "${@:2}"
}

for group in "${c2_groups[@]}"; do
  [[ -n "${selected[${group}]:-}" ]] || continue
  case "${group}" in
    c2-format)
      run_gate test-c2-format.sh
      run_gate test-c2-core.sh --core-only
      run_gate test-c2-checkpoint-inspect.sh
      run_gate test-c2-codec.sh
      run_gate test-c2-x1-canonical.sh
      run_gate test-c2-d2-cursor-pair.sh
      run_gate test-c2-persistence-policy.sh
      ;;
    c2-state)
      run_gate test-c2-training-state-owner.sh
      run_gate test-c2-model-encode.sh
      run_gate test-c2-o2-encode.sh
      ;;
    c2-save)
      run_gate test-c2-checkpoint-save.sh
      ;;
    c2-load)
      run_gate test-c2-checkpoint-load.sh --load-only
      ;;
    c2-public)
      run_gate test-c2-public.sh
      ;;
    c2-operational)
      run_gate test-c2-checkpoint-operational.sh --operational-only
      ;;
  esac
  printf 'C2 GROUP PASS: %s\n' "${group}"
done

if (( ${#selected[@]} == ${#c2_groups[@]} )); then
  printf 'C2 GATE PASS: private parser/codec/lifecycle, exact public aggregate, K2-authenticated load/save/release, and fixed-ceiling operational checkpoint\n'
fi
