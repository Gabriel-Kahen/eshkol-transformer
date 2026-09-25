#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp sha256sum; do
  require_command "${command}"
done

legacy_facade="${PROJECT_ROOT}/tests/c2/legacy_facades/transformer/trainer.esk"
[[ "$(sha256sum "${legacy_facade}" | awk '{print $1}')" == \
   b2f3818cc7645a9accf397479e3e4f859d73b635e39752310e76cd067921b9a8 ]] || \
  die "versioned C2 trainer facade changed"

artifact_dir="${1:-$(project_build_dir)/c2}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.c2-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

E1B_COMPILER_TIMEOUT_SECONDS="${C2_COMPILER_TIMEOUT_SECONDS:-900}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${PROJECT_ROOT}/native/c2_wave2_root.esk" \
    "${PROJECT_ROOT}/native/c2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/c2_wave2_private_renames.txt" \
    "${PROJECT_ROOT}/native/c2_wave2_public_exports.txt" \
    "${temporary_dir}/c2_wave2.o" \
    "${PROJECT_ROOT}/internal/p1/lib" \
    "${PROJECT_ROOT}/internal/c1/lib" \
    "${PROJECT_ROOT}/internal/t2/lib" \
    "${PROJECT_ROOT}/internal/t1/lib" \
    "${PROJECT_ROOT}/internal/d2/lib" \
    "${PROJECT_ROOT}/src"
cmp "${PROJECT_ROOT}/native/c2_wave2_defined_symbols.txt" \
  "${temporary_dir}/c2_wave2.o.evidence/global-defined.txt"
ar rcsD "${temporary_dir}/libeshkol_transformer_wave2.a" \
  "${temporary_dir}/c2_wave2.o"

mkdir -p "${artifact_dir}"
rm -rf -- "${artifact_dir}/c2_wave2.o.evidence"
mv "${temporary_dir}/c2_wave2.o.evidence" \
  "${artifact_dir}/c2_wave2.o.evidence"
mv -f "${temporary_dir}/c2_wave2.o" "${artifact_dir}/c2_wave2.o"
mv -f "${temporary_dir}/libeshkol_transformer_wave2.a" \
  "${artifact_dir}/libeshkol_transformer_wave2.a"
mkdir -p "${artifact_dir}/facades/transformer"
cp "${legacy_facade}" \
  "${artifact_dir}/facades/transformer/trainer.esk"
printf 'built source-composed C2 successor aggregate: %s\n' \
  "${artifact_dir}/libeshkol_transformer_wave2.a"
