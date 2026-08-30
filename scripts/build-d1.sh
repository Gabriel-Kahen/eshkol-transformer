#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/d1}"
mode="${2:-normal}"

case "${mode}" in
  normal)
    private_root="${PROJECT_ROOT}/native/d1_e1b_private.esk"
    library_name=libeshkol_transformer_d1.a
    ;;
  test-faults)
    private_root="${PROJECT_ROOT}/tests/d1/d1_e1b_fault_root.esk"
    library_name=libeshkol_transformer_d1_faults.a
    [[ "$(realpath -m -- "${artifact_dir}")" != \
       "$(realpath -m -- "$(project_build_dir)/d1")" ]] || \
      die "D1 test-fault artifact cannot target the canonical production directory"
    ;;
  *) die "unknown D1 build mode: ${mode}" ;;
esac

parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.d1-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

unset E1B_PACKAGE_POLICY
"${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${private_root}" \
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
