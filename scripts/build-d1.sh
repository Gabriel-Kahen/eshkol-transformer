#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/d1}"
mode="${2:-normal}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.d1-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

case "${mode}" in
  normal)
    package_policy=d1
    library_name=libeshkol_transformer_d1.a
    ;;
  test-faults)
    package_policy=d1-test-faults
    library_name=libeshkol_transformer_d1_faults.a
    ;;
  *) die "unknown D1 build mode: ${mode}" ;;
esac

E1B_PACKAGE_POLICY="${package_policy}" \
  "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${PROJECT_ROOT}/native/d1_e1b_private.esk" \
    "${PROJECT_ROOT}/native/d1_e1b_package_bridge.c" \
    "${PROJECT_ROOT}/native/d1_e1b_private_renames.txt" \
    "${PROJECT_ROOT}/native/d1_e1b_public_exports.txt" \
    "${temporary_dir}/stdlib.o" "${PROJECT_ROOT}/src"
ar rcsD "${temporary_dir}/${library_name}" "${temporary_dir}/stdlib.o"

mkdir -p "${artifact_dir}"
rm -f -- "${artifact_dir}/stdlib.o" "${artifact_dir}/stdlib.d" \
  "${artifact_dir}/data_io.o"
rm -rf -- "${artifact_dir}/stdlib.o.evidence"
mv -f "${temporary_dir}/${library_name}" "${artifact_dir}/${library_name}"
rm -rf -- "${artifact_dir}/${library_name}.evidence"
mv "${temporary_dir}/stdlib.o.evidence" \
  "${artifact_dir}/${library_name}.evidence"
printf 'built D1 combined E1B artifact: %s\n' \
  "${artifact_dir}/${library_name}"
