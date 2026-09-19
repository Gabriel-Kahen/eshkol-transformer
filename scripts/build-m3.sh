#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# -le 1 ]] || die "usage: $0 [ARTIFACT_DIR]"
# Reject predecessor-pin drift before compiling any canonical or fresh package.
(cd "${PROJECT_ROOT}" && python3 -m unittest -q \
  tests.m3.test_package_contract.PackageContract.test_predecessor_pin_inventory)
verify_toolchain
for command in ar cmp; do require_command "${command}"; done
m3_artifact_dir="${1:-$(project_build_dir)/m3}"
mkdir -p "$(dirname -- "${m3_artifact_dir}")"
m3_tmp="$(mktemp -d "$(dirname -- "${m3_artifact_dir}")/.m3-build.XXXXXX")"
trap 'rm -rf -- "${m3_tmp}"' EXIT
E1B_COMPILER_TIMEOUT_SECONDS="${M3_COMPILER_TIMEOUT_SECONDS:-600}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/m3_package_root.esk" \
  "${PROJECT_ROOT}/native/m3_package_bridge.c" \
  "${PROJECT_ROOT}/native/m3_package_private_renames.txt" \
  "${PROJECT_ROOT}/native/m3_package_public_exports.txt" \
  "${m3_tmp}/m3_package.o" \
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
cmp "${PROJECT_ROOT}/native/m3_package_defined_symbols.txt" \
  "${m3_tmp}/m3_package.o.evidence/global-defined.txt"
ar rcsD "${m3_tmp}/libeshkol_transformer_m3.a" "${m3_tmp}/m3_package.o"
ar t "${m3_tmp}/libeshkol_transformer_m3.a" >"${m3_tmp}/archive-members.txt"
cmp "${PROJECT_ROOT}/native/m3_package_archive_members.txt" "${m3_tmp}/archive-members.txt"
mkdir -p "${m3_tmp}/facades/transformer"
printf 'transformer/%s.esk\n' config diagnostic_transport error_consumer error_public model module tokenizer \
  >"${m3_tmp}/facades.txt"
cmp "${PROJECT_ROOT}/native/m3_package_facades.txt" "${m3_tmp}/facades.txt"
while IFS= read -r m3_facade; do
  [[ "$(realpath -- "${PROJECT_ROOT}/lib/${m3_facade}")" == "${PROJECT_ROOT}/lib/${m3_facade}" ]] || \
    die "M3 installation rejects symlinked public facade inputs"
  cp "${PROJECT_ROOT}/lib/${m3_facade}" "${m3_tmp}/facades/${m3_facade}"
done <"${m3_tmp}/facades.txt"
mkdir -p "${m3_artifact_dir}"
# Publish only after all exact artifact checks succeed.
for m3_output in m3_package.o m3_package.o.evidence libeshkol_transformer_m3.a facades; do
  rm -rf -- "${m3_artifact_dir:?}/${m3_output}"
  mv "${m3_tmp}/${m3_output}" "${m3_artifact_dir}/${m3_output}"
done
printf 'built restricted M3 diagnostic sibling: %s\n' "${m3_artifact_dir}/libeshkol_transformer_m3.a"
