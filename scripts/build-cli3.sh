#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp; do
  require_command "${command}"
done

artifact_dir="${1:-$(project_build_dir)/cli3}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.cli3-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

E1B_COMPILER_TIMEOUT_SECONDS="${CLI3_COMPILER_TIMEOUT_SECONDS:-900}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${PROJECT_ROOT}/native/cli3_root.esk" \
    "${PROJECT_ROOT}/native/cli3_package_bridge.c" \
    "${PROJECT_ROOT}/native/cli3_private_renames.txt" \
    "${PROJECT_ROOT}/native/cli3_public_exports.txt" \
    "${temporary_dir}/cli3.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t2/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/internal/d2/lib" \
    "${PROJECT_ROOT}/src"

ar rcsD "${temporary_dir}/libeshkol_transformer_cli3.a" \
  "${temporary_dir}/cli3.o"
cmp "${PROJECT_ROOT}/native/cli3_defined_symbols.txt" \
  "${temporary_dir}/cli3.o.evidence/global-defined.txt"

runner="$(eshkol_build_dir)/eshkol-run"
cxx="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cxx_path)"
mkdir -p "${temporary_dir}/app-cache"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${temporary_dir}/app-cache" \
  ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  "${runner}" --strict-types --no-stdlib -I "${PROJECT_ROOT}/lib" \
    -L "${temporary_dir}" --lib eshkol_transformer_cli3 \
    "${PROJECT_ROOT}/src/eshkol_transformer/cli.esk" \
    -o "${temporary_dir}/eshkol-transformer"

mkdir -p "${artifact_dir}"
rm -rf -- "${artifact_dir}/cli3.o.evidence"
mv "${temporary_dir}/cli3.o.evidence" "${artifact_dir}/cli3.o.evidence"
mv -f "${temporary_dir}/cli3.o" "${artifact_dir}/cli3.o"
mv -f "${temporary_dir}/libeshkol_transformer_cli3.a" \
  "${artifact_dir}/libeshkol_transformer_cli3.a"
mv -f "${temporary_dir}/eshkol-transformer" \
  "${artifact_dir}/eshkol-transformer"
printf 'built source-composed CLI3 C2 successor: %s\n' \
  "${artifact_dir}/libeshkol_transformer_cli3.a"
