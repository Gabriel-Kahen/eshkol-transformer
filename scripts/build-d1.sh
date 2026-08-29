#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/d1}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.d1-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC)
library_name=libeshkol_transformer_d1.a
case "${mode}" in
  normal) ;;
  test-faults)
    cflags+=(-DET_D1_TEST_FAULTS)
    library_name=libeshkol_transformer_d1_faults.a
    ;;
  *) die "unknown D1 build mode: ${mode}" ;;
esac

"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/data_io.c" \
  -o "${temporary_dir}/data_io.o"
ar rcsD "${temporary_dir}/${library_name}" "${temporary_dir}/data_io.o"

mkdir -p "${artifact_dir}"
mv -f "${temporary_dir}/data_io.o" "${artifact_dir}/data_io.o"
mv -f "${temporary_dir}/${library_name}" "${artifact_dir}/${library_name}"
printf 'built D1 native I/O: %s\n' "${artifact_dir}/${library_name}"
