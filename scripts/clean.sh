#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_dir="$(project_build_dir)"
[[ "${build_dir}" == "${PROJECT_ROOT}/build" ]] || die "refusing to clean non-default directory: ${build_dir}"
if [[ -d "${build_dir}" ]]; then
  find "${build_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
fi
printf 'cleaned: %s (toolchain artifacts preserved)\n' "${build_dir}"
