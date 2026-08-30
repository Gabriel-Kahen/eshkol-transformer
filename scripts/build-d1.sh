#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
require_command objcopy

artifact_dir="${1:-$(project_build_dir)/d1}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
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

env XDG_CACHE_HOME="${temporary_dir}/cache" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  "${runner}" --strict-types --no-stdlib \
  -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/src" \
  --shared-lib --compile-only \
  --emit-depfile "${temporary_dir}/stdlib.d" \
  "${PROJECT_ROOT}/lib/stdlib.esk" -o "${temporary_dir}/stdlib.o"

private_bindings=(
  d1-compiled-public-operations
)
objcopy_args=()
private_index=0
for private_binding in "${private_bindings[@]}"; do
  objcopy_args+=(--localize-symbol="${private_binding}")
  objcopy_args+=(--redefine-sym="${private_binding}=.Lprivate_slot_${private_index}")
  private_index=$((private_index + 1))
done
objcopy "${objcopy_args[@]}" "${temporary_dir}/stdlib.o"

mkdir -p "${artifact_dir}"
mv -f "${temporary_dir}/data_io.o" "${artifact_dir}/data_io.o"
mv -f "${temporary_dir}/${library_name}" "${artifact_dir}/${library_name}"
mv -f "${temporary_dir}/stdlib.o" "${artifact_dir}/stdlib.o"
mv -f "${temporary_dir}/stdlib.d" "${artifact_dir}/stdlib.d"
printf 'built D1 precompiled module: %s\n' "${artifact_dir}/stdlib.o"
printf 'built D1 checked native primitive: %s\n' "${artifact_dir}/${library_name}"
