#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 1 ]] || die "usage: $0 [ARTIFACT_DIR]"
verify_toolchain
for command in ar cmp; do require_command "${command}"; done
m3t_artifact_dir="${1:-$(project_build_dir)/m3t}"
mkdir -p "$(dirname -- "${m3t_artifact_dir}")"
m3t_tmp="$(mktemp -d "$(dirname -- "${m3t_artifact_dir}")/.m3t-build.XXXXXX")"
trap 'rm -rf -- "${m3t_tmp}"' EXIT
E1B_COMPILER_TIMEOUT_SECONDS="${M3T_COMPILER_TIMEOUT_SECONDS:-600}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/m3t_package_root.esk" \
  "${PROJECT_ROOT}/native/m3t_package_bridge.c" \
  "${PROJECT_ROOT}/native/m3t_package_private_renames.txt" \
  "${PROJECT_ROOT}/native/m3t_package_public_exports.txt" \
  "${m3t_tmp}/m3t_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
cmp "${PROJECT_ROOT}/native/m3t_package_defined_symbols.txt" \
  "${m3t_tmp}/m3t_package.o.evidence/global-defined.txt"
ar rcsD "${m3t_tmp}/libeshkol_transformer_m3t.a" "${m3t_tmp}/m3t_package.o"
ar t "${m3t_tmp}/libeshkol_transformer_m3t.a" >"${m3t_tmp}/archive-members.txt"
cmp "${PROJECT_ROOT}/native/m3t_package_archive_members.txt" "${m3t_tmp}/archive-members.txt"
mkdir -p "${m3t_tmp}/facades/transformer"
printf 'transformer/%s.esk\n' config diagnostic_transport error_consumer error_public module tokenizer \
  >"${m3t_tmp}/facades.txt"
cmp "${PROJECT_ROOT}/native/m3t_package_facades.txt" "${m3t_tmp}/facades.txt"
while IFS= read -r m3t_facade; do
  [[ "$(realpath -- "${PROJECT_ROOT}/lib/${m3t_facade}")" == "${PROJECT_ROOT}/lib/${m3t_facade}" ]] || \
    die "M3T installation rejects symlinked public facade inputs"
  cp "${PROJECT_ROOT}/lib/${m3t_facade}" "${m3t_tmp}/facades/${m3t_facade}"
done <"${m3t_tmp}/facades.txt"
mkdir -p "${m3t_artifact_dir}"
# Publish only after all exact artifact checks succeed.
for m3t_output in m3t_package.o m3t_package.o.evidence libeshkol_transformer_m3t.a facades; do
  rm -rf -- "${m3t_artifact_dir:?}/${m3t_output}"
  mv "${m3t_tmp}/${m3t_output}" "${m3t_artifact_dir}/${m3t_output}"
done
printf 'built restricted M3T diagnostic sibling: %s\n' "${m3t_artifact_dir}/libeshkol_transformer_m3t.a"
