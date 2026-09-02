#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

artifact_dir="${1:-$(project_build_dir)/a2}"
mode="${2:-normal}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent_dir="$(dirname -- "${artifact_dir}")"
mkdir -p "${parent_dir}"
temporary_dir="$(mktemp -d "${parent_dir}/.a2-build.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fPIC -I "${PROJECT_ROOT}/include"
)
case "${mode}" in
  normal)
    cflags+=(-O2)
    ;;
  sanitize)
    cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    ;;
  *) die "unknown A2 build mode: ${mode}" ;;
esac

"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/a2_attention_provider.c" \
  -o "${temporary_dir}/a2_attention_provider.o"
"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/a2_kv_cache.c" \
  -o "${temporary_dir}/a2_kv_cache.o"
ar rcsD "${temporary_dir}/libeshkol_transformer_a2.a" \
  "${temporary_dir}/a2_attention_provider.o" \
  "${temporary_dir}/a2_kv_cache.o"

mkdir -p "${artifact_dir}"
mv -f "${temporary_dir}/a2_attention_provider.o" "${artifact_dir}/a2_attention_provider.o"
mv -f "${temporary_dir}/a2_kv_cache.o" "${artifact_dir}/a2_kv_cache.o"
mv -f "${temporary_dir}/libeshkol_transformer_a2.a" \
  "${artifact_dir}/libeshkol_transformer_a2.a"
printf 'built A2 provider/cache ABI: %s\n' "${artifact_dir}/libeshkol_transformer_a2.a"
