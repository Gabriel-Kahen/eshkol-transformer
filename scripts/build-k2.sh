#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp; do
  require_command "${command}"
done

artifact_dir="${1:-$(project_build_dir)/k2}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.k2-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

E1B_COMPILER_TIMEOUT_SECONDS="${K2_COMPILER_TIMEOUT_SECONDS:-360}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/k2_wave2_root.esk" \
  "${PROJECT_ROOT}/native/k2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/k2_wave2_private_renames.txt" \
  "${PROJECT_ROOT}/native/k2_wave2_public_exports.txt" \
  "${temporary_dir}/k2_wave2.o" \
  "${PROJECT_ROOT}/internal/p1/lib" \
  "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" \
  "${PROJECT_ROOT}/src"
cmp "${PROJECT_ROOT}/native/k2_wave2_defined_symbols.txt" \
  "${temporary_dir}/k2_wave2.o.evidence/global-defined.txt"
ar rcsD "${temporary_dir}/libeshkol_transformer_wave2.a" \
  "${temporary_dir}/k2_wave2.o"

mkdir -p "${artifact_dir}"
rm -rf -- "${artifact_dir}/k2_wave2.o.evidence"
mv "${temporary_dir}/k2_wave2.o.evidence" \
  "${artifact_dir}/k2_wave2.o.evidence"
mv -f "${temporary_dir}/k2_wave2.o" "${artifact_dir}/k2_wave2.o"
mv -f "${temporary_dir}/libeshkol_transformer_wave2.a" \
  "${artifact_dir}/libeshkol_transformer_wave2.a"
printf 'built source-composed K2 Wave 2 aggregate: %s\n' \
  "${artifact_dir}/libeshkol_transformer_wave2.a"
