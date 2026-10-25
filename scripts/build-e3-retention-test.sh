#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 3 ]] || die "usage: $0 [ARTIFACT_DIR] [normal|sanitize] [retention|diagnostic]"
verify_toolchain
for command in ar cmp; do require_command "${command}"; done
artifact="${1:-$(project_build_dir)/e3-retention-test}"
mode="${2:-normal}"
case "${mode}" in normal|sanitize) ;; *) die "unknown E3 retention build mode: ${mode}" ;; esac
tuple="${3:-retention}"
case "${tuple}" in retention|diagnostic) ;; *) die "unknown E3 retention tuple: ${tuple}" ;; esac
mkdir -p "$(dirname -- "${artifact}")"
staging="$(mktemp -d "$(dirname -- "${artifact}")/.e3-retention.XXXXXX")"
source "${PROJECT_ROOT}/scripts/e3-mode-graph-stage-cleanup.sh"
e3_mode_graph_install_stage_cleanup "${staging}" retention
prefix="${PROJECT_ROOT}/tests/e3_private/retention_package"
driver_root="${PROJECT_ROOT}/tests/e3_private/retention_driver_root.esk"
bridge="${PROJECT_ROOT}/tests/e3_private/retention_bridge.c"
if [[ "${tuple}" == diagnostic ]]; then
  prefix="${PROJECT_ROOT}/tests/e3_private/retention_arena_diagnostic_package"
  driver_root="${PROJECT_ROOT}/tests/e3_private/retention_arena_diagnostic_driver_root.esk"
  bridge="${PROJECT_ROOT}/tests/e3_private/retention_arena_diagnostic_bridge.c"
fi
sanitize=0
[[ "${mode}" != sanitize ]] || sanitize=1
E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
  E3_RETENTION_SANITIZE="${sanitize}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${driver_root}" \
  "${bridge}" \
  "${prefix}_private_renames.txt" \
  "${prefix}_public_exports.txt" \
  "${staging}/e3_retention_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
cmp "${prefix}_defined_symbols.txt" \
  "${staging}/e3_retention_package.o.evidence/global-defined.txt"
ar rcsD "${staging}/libeshkol_transformer_e3_retention.a" \
  "${staging}/e3_retention_package.o"
ar t "${staging}/libeshkol_transformer_e3_retention.a" | \
  cmp "${prefix}_archive_members.txt" -
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${staging}" "${artifact}"
printf 'built closed E3 retention test artifact: %s\n' \
  "${artifact}/libeshkol_transformer_e3_retention.a"
