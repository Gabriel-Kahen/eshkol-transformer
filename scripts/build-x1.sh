#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/x1}"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.x1-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

combined_object="${temporary_dir}/x1_consumer.o"
E1B_COMPILER_TIMEOUT_SECONDS="${E1B_COMPILER_TIMEOUT_SECONDS:-240}" \
  "${PROJECT_ROOT}/scripts/build-e1b-consumer.sh" \
    "${PROJECT_ROOT}/native/x1_config_consumer_root.esk" \
    "${PROJECT_ROOT}/native/x1_config_consumer_bridge.c" \
    "${PROJECT_ROOT}/native/x1_config_private_renames.txt" \
    "${PROJECT_ROOT}/native/x1_config_public_exports.txt" \
    "${combined_object}" "${PROJECT_ROOT}/native"

ar rcsD "${temporary_dir}/libeshkol_transformer_x1.a" \
  "${combined_object}"

mkdir -p "${artifact_dir}"
rm -rf -- "${artifact_dir}/x1_consumer.o.evidence"
mv -f "${combined_object}" "${artifact_dir}/x1_consumer.o"
mv -f "${combined_object}.evidence" \
  "${artifact_dir}/x1_consumer.o.evidence"
mv -f "${temporary_dir}/libeshkol_transformer_x1.a" \
  "${artifact_dir}/libeshkol_transformer_x1.a"
printf 'built X1 configuration consumer: %s\n' \
  "${artifact_dir}/libeshkol_transformer_x1.a"
