#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 2 ]] || die "usage: $0 [ARTIFACT_DIR] [normal|wrong-target|mean-of-means|second-normalization|wrong-vocab-axis|physical-capacity|sanitize]"
verify_toolchain
for command in ar cmp; do require_command "${command}"; done
artifact="${1:-$(project_build_dir)/e3-numerical-test}"
mode="${2:-normal}"
case "${mode}" in normal|wrong-target|mean-of-means|second-normalization|wrong-vocab-axis|physical-capacity|sanitize) ;; *) die "unknown E3 numerical mode: ${mode}" ;; esac
mkdir -p "$(dirname -- "${artifact}")"
staging="$(mktemp -d "$(dirname -- "${artifact}")/.e3-numerical.XXXXXX")"
numerical_stage_cleanup() {
  local status=$?
  if [[ "${status}" == 0 ]]; then
    rm -rf -- "${staging}"
  else
    printf 'E3 numerical rejected-build stage retained at %s\n' "${staging}" >&2
  fi
  return "${status}"
}
trap numerical_stage_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
prefix="${PROJECT_ROOT}/tests/e3_private/numerical_package"
mutant="${mode}"
sanitize=0
if [[ "${mode}" == sanitize ]]; then mutant=normal; sanitize=1; fi
E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
  E3_NUMERICAL_MUTANT="${mutant}" E3_NUMERICAL_SANITIZE="${sanitize}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/tests/e3_private/numerical_driver_root.esk" \
  "${PROJECT_ROOT}/tests/e3_private/numerical_bridge.c" \
  "${prefix}_private_renames.txt" \
  "${prefix}_public_exports.txt" \
  "${staging}/e3_numerical_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
cmp "${prefix}_defined_symbols.txt" \
  "${staging}/e3_numerical_package.o.evidence/global-defined.txt"
ar rcsD "${staging}/libeshkol_transformer_e3_numerical.a" \
  "${staging}/e3_numerical_package.o"
ar t "${staging}/libeshkol_transformer_e3_numerical.a" | \
  cmp "${prefix}_archive_members.txt" -
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${staging}" "${artifact}"
printf 'built closed E3 numerical test artifact: %s\n' \
  "${artifact}/libeshkol_transformer_e3_numerical.a"
