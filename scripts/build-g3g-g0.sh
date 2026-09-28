#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 1 ]] || die "usage: $0 [ARTIFACT_DIR]"
verify_toolchain
for command in ar cmp; do require_command "${command}"; done
g3g_g0_artifact_dir="${1:-$(project_build_dir)/g3g}"
mkdir -p "$(dirname -- "${g3g_g0_artifact_dir}")"
g3g_g0_tmp="$(mktemp -d "$(dirname -- "${g3g_g0_artifact_dir}")/.g3g-build.XXXXXX")"
trap 'rm -rf -- "${g3g_g0_tmp}"' EXIT
E1B_COMPILER_TIMEOUT_SECONDS="${G3G_G0_COMPILER_TIMEOUT_SECONDS:-1200}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/g3g_g0_package_root.esk" \
  "${PROJECT_ROOT}/native/g3g_g0_package_bridge.c" \
  "${PROJECT_ROOT}/native/g3g_g0_package_private_renames.txt" \
  "${PROJECT_ROOT}/native/g3g_g0_package_public_exports.txt" \
  "${g3g_g0_tmp}/g3g_g0_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
cmp "${PROJECT_ROOT}/native/g3g_g0_package_defined_symbols.txt" \
  "${g3g_g0_tmp}/g3g_g0_package.o.evidence/global-defined.txt"
ar rcsD "${g3g_g0_tmp}/libeshkol_transformer_g3g_g0.a" "${g3g_g0_tmp}/g3g_g0_package.o"
ar t "${g3g_g0_tmp}/libeshkol_transformer_g3g_g0.a" >"${g3g_g0_tmp}/archive-members.txt"
cmp "${PROJECT_ROOT}/native/g3g_g0_package_archive_members.txt" \
  "${g3g_g0_tmp}/archive-members.txt"
mkdir -p "${g3g_g0_tmp}/facades/transformer"
cp "${PROJECT_ROOT}/native/g3g_g0_package_facades.txt" "${g3g_g0_tmp}/facades.txt"
while IFS= read -r g3g_facade; do
  g3g_source="${PROJECT_ROOT}/lib/${g3g_facade}"
  [[ "$(realpath -- "${g3g_source}")" == "${g3g_source}" ]] || \
    die "G3-G installation rejects symlinked facade input"
  cp "${g3g_source}" "${g3g_g0_tmp}/facades/${g3g_facade}"
done <"${g3g_g0_tmp}/facades.txt"
mkdir -p "${g3g_g0_artifact_dir}"
for g3g_output in g3g_g0_package.o g3g_g0_package.o.evidence \
    libeshkol_transformer_g3g_g0.a facades; do
  rm -rf -- "${g3g_g0_artifact_dir:?}/${g3g_output}"
  mv "${g3g_g0_tmp}/${g3g_output}" "${g3g_g0_artifact_dir}/${g3g_output}"
done
printf 'built G3-G G0/G1 package: %s\n' \
  "${g3g_g0_artifact_dir}/libeshkol_transformer_g3g_g0.a"
