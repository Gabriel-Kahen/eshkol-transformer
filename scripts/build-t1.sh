#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp nm readelf; do
  require_command "${command}"
done

artifact_dir="${1:-$(project_build_dir)/t1}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.t1-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

aggregate_object="${temporary_dir}/wave1.o"
E1B_COMPILER_TIMEOUT_SECONDS="${T1_COMPILER_TIMEOUT_SECONDS:-300}" \
  "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/t1_wave1_root.esk" \
  "${PROJECT_ROOT}/native/t1_wave1_package_bridge.c" \
  "${PROJECT_ROOT}/native/t1_wave1_private_renames.txt" \
  "${PROJECT_ROOT}/native/t1_wave1_public_exports.txt" \
  "${aggregate_object}" \
  "${PROJECT_ROOT}/internal/p1/lib" \
  "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" \
  "${PROJECT_ROOT}/src"

[[ "$(wc -l < "${aggregate_object}.evidence/global-defined.txt")" == 47 ]] || \
  die "T1 aggregate must expose exactly 47 global definitions"
cmp "${PROJECT_ROOT}/native/t1_wave1_defined_symbols.txt" \
  "${aggregate_object}.evidence/global-defined.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_undefined_symbols.txt" \
  "${aggregate_object}.evidence/undefined.txt"

ar rcsD "${temporary_dir}/libeshkol_transformer_wave1.a" \
  "${aggregate_object}"

mkdir -p "${artifact_dir}"
rm -rf -- "${artifact_dir}/wave1.o.evidence.tmp"
mv "${aggregate_object}.evidence" \
  "${artifact_dir}/wave1.o.evidence.tmp"
mv -f "${aggregate_object}" "${artifact_dir}/wave1.o.tmp"
mv -f "${temporary_dir}/libeshkol_transformer_wave1.a" \
  "${artifact_dir}/libeshkol_transformer_wave1.a.tmp"
rm -rf -- "${artifact_dir}/wave1.o.evidence"
mv "${artifact_dir}/wave1.o.evidence.tmp" \
  "${artifact_dir}/wave1.o.evidence"
mv -f "${artifact_dir}/wave1.o.tmp" "${artifact_dir}/wave1.o"
mv -f "${artifact_dir}/libeshkol_transformer_wave1.a.tmp" \
  "${artifact_dir}/libeshkol_transformer_wave1.a"

printf 'built canonical Wave 1 aggregate: %s\n' \
  "${artifact_dir}/libeshkol_transformer_wave1.a"
