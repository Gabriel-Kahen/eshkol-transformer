#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/n2}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.n2-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC -I "${PROJECT_ROOT}/include"
)
case "${mode}" in
  normal)
    cflags+=(-O2)
    ;;
  sanitize)
    cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    ;;
  *) die "unknown N2 build mode: ${mode}" ;;
esac

"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/native/n2_primitives_provider.c" \
  -o "${temporary_dir}/n2_primitives_provider.o"
ar rcsD "${temporary_dir}/libeshkol_transformer_n2.a" \
  "${temporary_dir}/n2_primitives_provider.o"

mkdir -p "${artifact_dir}"
mv -f "${temporary_dir}/n2_primitives_provider.o" \
  "${artifact_dir}/n2_primitives_provider.o"
mv -f "${temporary_dir}/libeshkol_transformer_n2.a" \
  "${artifact_dir}/libeshkol_transformer_n2.a"
printf 'built N2 carrier-neutral provider: %s\n' \
  "${artifact_dir}/libeshkol_transformer_n2.a"
