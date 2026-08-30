#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/i1}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.i1-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fPIC -I "${PROJECT_ROOT}/include"
)
case "${mode}" in
  normal) ;;
  sanitize-test)
    cflags+=(
      -DET_I64_TENSOR_TESTING
      -fsanitize=address,undefined -fno-omit-frame-pointer
    )
    ;;
  *) die "unknown I1 build mode: ${mode}" ;;
esac

"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/i64_tensor.c" \
  -o "${temporary_dir}/i64_tensor.o"
ar rcsD "${temporary_dir}/libeshkol_transformer_i64.a" \
  "${temporary_dir}/i64_tensor.o"

mkdir -p "${artifact_dir}"
mv -f "${temporary_dir}/i64_tensor.o" "${artifact_dir}/i64_tensor.o"
mv -f "${temporary_dir}/libeshkol_transformer_i64.a" \
  "${artifact_dir}/libeshkol_transformer_i64.a"
printf 'built I1 exact-i64 container: %s\n' \
  "${artifact_dir}/libeshkol_transformer_i64.a"
