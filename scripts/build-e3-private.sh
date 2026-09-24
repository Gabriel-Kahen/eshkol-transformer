#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 1 ]] || die "usage: $0 [ARTIFACT_DIR]"
verify_toolchain
for command in ar cmp python3; do require_command "${command}"; done
e3_artifact_dir="${1:-$(project_build_dir)/e3-private}"
mkdir -p "$(dirname -- "${e3_artifact_dir}")"
e3_tmp="$(mktemp -d "$(dirname -- "${e3_artifact_dir}")/.e3-private-build.XXXXXX")"
trap 'rm -rf -- "${e3_tmp}"' EXIT
E1B_COMPILER_TIMEOUT_SECONDS="${E3_COMPILER_TIMEOUT_SECONDS:-900}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/e3_private_driver_root.esk" \
  "${PROJECT_ROOT}/native/e3_private_bridge.c" \
  "${PROJECT_ROOT}/native/e3_private_package_private_renames.txt" \
  "${PROJECT_ROOT}/native/e3_private_package_public_exports.txt" \
  "${e3_tmp}/e3_private_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
cmp "${PROJECT_ROOT}/native/e3_private_package_defined_symbols.txt" \
  "${e3_tmp}/e3_private_package.o.evidence/global-defined.txt"
ar rcsD "${e3_tmp}/libeshkol_transformer_e3_private.a" \
  "${e3_tmp}/e3_private_package.o"
ar t "${e3_tmp}/libeshkol_transformer_e3_private.a" \
  >"${e3_tmp}/archive-members.txt"
cmp "${PROJECT_ROOT}/native/e3_private_package_archive_members.txt" \
  "${e3_tmp}/archive-members.txt"
cp "${e3_tmp}/e3_private_package.o.evidence/d2-generation.json" \
  "${e3_tmp}/d2-generation.json"
# A replacement exchanges the complete old and new generations in one namespace
# operation; the EXIT trap then removes the old generation at the staging path.
python3 "${PROJECT_ROOT}/scripts/e3-atomic-publish.py" \
  "${e3_tmp}" "${e3_artifact_dir}"
printf 'built closed E3 private artifact: %s\n' \
  "${e3_artifact_dir}/libeshkol_transformer_e3_private.a"
