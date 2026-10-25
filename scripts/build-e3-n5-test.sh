#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 2 ]] || die "usage: $0 [ARTIFACT_DIR] [normal|sanitize]"
verify_toolchain
for command in ar cmp python3; do require_command "${command}"; done
artifact="${1:-$(project_build_dir)/e3-n5-test}"
mode="${2:-normal}"
case "${mode}" in normal|sanitize) ;; *) die "unknown E3 N5 build mode" ;; esac
mkdir -p "$(dirname -- "${artifact}")"
stage="$(mktemp -d "$(dirname -- "${artifact}")/.e3-n5.XXXXXX")"
n5_stage_cleanup() {
  local status=$?
  if [[ "${status}" == 0 ]]; then
    rm -rf -- "${stage}"
  else
    printf 'E3 N5 rejected build retained at %s\n' "${stage}" >&2
  fi
  return "${status}"
}
trap n5_stage_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
prefix="${PROJECT_ROOT}/tests/e3_n5/package"
sanitize=0
[[ "${mode}" != sanitize ]] || sanitize=1
E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
  E3_N5_SANITIZE="${sanitize}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/tests/e3_n5/driver_root.esk" \
  "${PROJECT_ROOT}/tests/e3_n5/bridge.c" \
  "${prefix}_private_renames.txt" "${prefix}_public_exports.txt" \
  "${stage}/e3_n5_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
cmp "${prefix}_defined_symbols.txt" \
  "${stage}/e3_n5_package.o.evidence/global-defined.txt"
ar rcsD "${stage}/libeshkol_transformer_e3_n5.a" \
  "${stage}/e3_n5_package.o"
ar t "${stage}/libeshkol_transformer_e3_n5.a" | \
  cmp "${prefix}_archive_members.txt" -
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${stage}" "${artifact}"
printf 'built closed E3 N5 test artifact: %s\n' \
  "${artifact}/libeshkol_transformer_e3_n5.a"
