#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/d2}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.d2-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
case "${mode}" in
  normal) ;;
  test)
    cflags+=(-DET_D2_NATIVE_TESTING -DET_I64_TENSOR_TESTING)
    ;;
  sanitize-test)
    cflags+=(
      -DET_D2_NATIVE_TESTING -DET_I64_TENSOR_TESTING
      -fsanitize=address,undefined -fno-omit-frame-pointer
    )
    ;;
  *) die "unknown D2 build mode: ${mode}" ;;
esac

"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/d2_native.c" \
  -o "${temporary_dir}/d2_native.o"
ar rcsD "${temporary_dir}/libeshkol_transformer_d2_private.a" \
  "${temporary_dir}/d2_native.o"

mkdir -p "${artifact_dir}"
mv -f "${temporary_dir}/d2_native.o" "${artifact_dir}/d2_native.o"
mv -f "${temporary_dir}/libeshkol_transformer_d2_private.a" \
  "${artifact_dir}/libeshkol_transformer_d2_private.a"
printf 'built D2 private carrier candidate: %s\n' \
  "${artifact_dir}/libeshkol_transformer_d2_private.a"
