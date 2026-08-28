#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
build_dir="$(project_build_dir)"
mkdir -p "${build_dir}"
source_commit="$(git -C "$(eshkol_source_dir)" rev-parse HEAD)"
llvm_version="$("$(llvm_config)" --version)"
compiler_sha256="$(sha256sum "$(eshkol_build_dir)/eshkol-run" | awk '{ print $1 }')"
host_supported=true
if [[ "${ESHKOL_ALLOW_UNSUPPORTED_HOST:-0}" == 1 ]]; then
  host_supported=false
fi

{
  printf 'format_version\t1\n'
  printf 'eshkol_repository\t%s\n' "$(lock_value eshkol_repository)"
  printf 'eshkol_commit\t%s\n' "${source_commit}"
  printf 'eshkol_version\t%s\n' "$(lock_value eshkol_version)"
  printf 'eshkol_binary_sha256\t%s\n' "${compiler_sha256}"
  printf 'llvm_version\t%s\n' "${llvm_version}"
  printf 'host_supported\t%s\n' "${host_supported}"
  printf 'host_uname\t%s\n' "$(uname -srm)"
} > "${build_dir}/toolchain-manifest.tsv"
printf 'configured: %s\n' "${build_dir}/toolchain-manifest.tsv"
