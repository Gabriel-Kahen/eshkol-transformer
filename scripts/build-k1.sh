#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/k1}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.k1-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fPIC -I "${PROJECT_ROOT}/include"
)
case "${mode}" in
  normal) ;;
  sanitize)
    cflags+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
    ;;
  *) die "unknown K1 build mode: ${mode}" ;;
esac

"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/kernel_abi.c" \
  -o "${temporary_dir}/kernel_abi.o"
ar rcsD "${temporary_dir}/libeshkol_transformer_k1.a" \
  "${temporary_dir}/kernel_abi.o"

mkdir -p "${artifact_dir}"
mv -f "${temporary_dir}/kernel_abi.o" "${artifact_dir}/kernel_abi.o"
mv -f "${temporary_dir}/libeshkol_transformer_k1.a" \
  "${artifact_dir}/libeshkol_transformer_k1.a"
printf 'built K1 ABI: %s\n' "${artifact_dir}/libeshkol_transformer_k1.a"
