#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 2 ]] || die "usage: $0 ARTIFACT_DIR normal|sanitize"
verify_toolchain
for command in ar cmp; do require_command "${command}"; done
artifact=$1
case "$2" in normal|sanitize) mode=$2 ;; *) die "invalid E3 stage-10 mode" ;; esac
[[ ! -e "$artifact" && ! -L "$artifact" ]] || die "artifact already exists"
mkdir -p "$(dirname -- "$artifact")"
staging="$(mktemp -d "$(dirname -- "$artifact")/.e3-fault.XXXXXX")"
source "${PROJECT_ROOT}/scripts/e3-mode-graph-stage-cleanup.sh"
e3_mode_graph_install_stage_cleanup "$staging" fault
prefix="${PROJECT_ROOT}/tests/e3_private/fault_package"
sanitize=0
[[ "$mode" != sanitize ]] || sanitize=1
E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
  E3_RETENTION_SANITIZE="$sanitize" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/tests/e3_private/fault_driver_root.esk" \
  "${PROJECT_ROOT}/tests/e3_private/fault_bridge.c" \
  "${prefix}_private_renames.txt" "${prefix}_public_exports.txt" \
  "${staging}/e3_fault_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
cmp "${prefix}_defined_symbols.txt" \
  "${staging}/e3_fault_package.o.evidence/global-defined.txt"
ar rcsD "${staging}/libeshkol_transformer_e3_r1_stage10.a" \
  "${staging}/e3_fault_package.o"
ar t "${staging}/libeshkol_transformer_e3_r1_stage10.a" | \
  cmp "${prefix}_archive_members.txt" -
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" "$staging" "$artifact"
printf 'built E3 R1 stage-10 fault package: %s\n' "$artifact"
