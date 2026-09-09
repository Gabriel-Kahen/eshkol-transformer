#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
n3k_artifact_dir="${1:-$(project_build_dir)/n3k}"
n3k_mode="${2:-normal}"
n3k_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
mkdir -p "$(dirname -- "${n3k_artifact_dir}")"
n3k_temporary_dir="$(mktemp -d "$(dirname -- "${n3k_artifact_dir}")/.n3k-build.XXXXXX")"
trap 'rm -rf -- "${n3k_temporary_dir}"' EXIT
n3k_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC -I "${PROJECT_ROOT}/include"
)
case "${n3k_mode}" in
  normal) n3k_cflags+=(-O2) ;;
  sanitize) n3k_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown N3K build mode: ${n3k_mode}" ;;
esac
"${n3k_cc}" "${n3k_cflags[@]}" \
  -MMD -MF "${n3k_temporary_dir}/n3k_primitives_provider.d" -MT n3k_primitives_provider.o \
  -c "${PROJECT_ROOT}/native/n3k_primitives_provider.c" \
  -o "${n3k_temporary_dir}/n3k_primitives_provider.o"
ar rcsD "${n3k_temporary_dir}/libeshkol_transformer_n3k.a" \
  "${n3k_temporary_dir}/n3k_primitives_provider.o"
mkdir -p "${n3k_artifact_dir}"
mv -f "${n3k_temporary_dir}/n3k_primitives_provider.o" "${n3k_artifact_dir}/"
mv -f "${n3k_temporary_dir}/n3k_primitives_provider.d" "${n3k_artifact_dir}/"
mv -f "${n3k_temporary_dir}/libeshkol_transformer_n3k.a" "${n3k_artifact_dir}/"
printf 'built N3K carrier-neutral provider: %s\n' \
  "${n3k_artifact_dir}/libeshkol_transformer_n3k.a"
