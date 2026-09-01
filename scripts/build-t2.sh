#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp grep nm readelf sed tr; do
  require_command "${command}"
done

artifact_dir="${1:-$(project_build_dir)/t2}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.t2-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

aggregate_object="${temporary_dir}/wave2.o"
E1B_COMPILER_TIMEOUT_SECONDS="${T2_COMPILER_TIMEOUT_SECONDS:-300}" \
  "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
  "${PROJECT_ROOT}/native/t2_wave2_root.esk" \
  "${PROJECT_ROOT}/native/t2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/t2_wave2_private_renames.txt" \
  "${PROJECT_ROOT}/native/t2_wave2_public_exports.txt" \
  "${aggregate_object}" \
  "${PROJECT_ROOT}/internal/p1/lib" \
  "${PROJECT_ROOT}/internal/c1/lib" \
  "${PROJECT_ROOT}/internal/t2/lib" \
  "${PROJECT_ROOT}/internal/t1/lib" \
  "${PROJECT_ROOT}/src"

[[ "$(wc -l < "${aggregate_object}.evidence/global-defined.txt")" == 46 ]] || \
  die "T2 aggregate must expose exactly the accepted 46 global definitions"
cmp "${PROJECT_ROOT}/native/t2_wave2_defined_symbols.txt" \
  "${aggregate_object}.evidence/global-defined.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_undefined_symbols.txt" \
  "${aggregate_object}.evidence/undefined.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_public_exports.txt" \
  "${PROJECT_ROOT}/native/t2_wave2_public_exports.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_defined_symbols.txt" \
  "${PROJECT_ROOT}/native/t2_wave2_defined_symbols.txt"
cmp "${PROJECT_ROOT}/native/t1_wave1_undefined_symbols.txt" \
  "${PROJECT_ROOT}/native/t2_wave2_undefined_symbols.txt"
sed -e 's/^[^:]*://' -e 's/\\//g' \
  "${aggregate_object}.evidence/private.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${temporary_dir}/source-closure.txt"
cmp "${PROJECT_ROOT}/native/t2_wave2_source_closure.txt" \
  "${temporary_dir}/source-closure.txt"

ar rcsD "${temporary_dir}/libeshkol_transformer_wave2.a" \
  "${aggregate_object}"

mkdir -p "${artifact_dir}"
rm -rf -- "${artifact_dir}/wave2.o.evidence.tmp"
mv "${aggregate_object}.evidence" \
  "${artifact_dir}/wave2.o.evidence.tmp"
mv -f "${aggregate_object}" "${artifact_dir}/wave2.o.tmp"
mv -f "${temporary_dir}/libeshkol_transformer_wave2.a" \
  "${artifact_dir}/libeshkol_transformer_wave2.a.tmp"
rm -rf -- "${artifact_dir}/wave2.o.evidence"
mv "${artifact_dir}/wave2.o.evidence.tmp" \
  "${artifact_dir}/wave2.o.evidence"
mv -f "${artifact_dir}/wave2.o.tmp" "${artifact_dir}/wave2.o"
mv -f "${artifact_dir}/libeshkol_transformer_wave2.a.tmp" \
  "${artifact_dir}/libeshkol_transformer_wave2.a"

printf 'built canonical Wave 2 aggregate: %s\n' \
  "${artifact_dir}/libeshkol_transformer_wave2.a"
