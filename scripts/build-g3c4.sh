#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
g3c4_artifact_dir="${1:-$(project_build_dir)/g3c4}"
g3c4_mode="${2:-normal}"
g3c4_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
mkdir -p "$(dirname -- "${g3c4_artifact_dir}")"
g3c4_temporary_dir="$(mktemp -d "$(dirname -- "${g3c4_artifact_dir}")/.g3c4-build.XXXXXX")"
trap 'rm -rf -- "${g3c4_temporary_dir}"' EXIT
g3c4_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC -I "${PROJECT_ROOT}/include"
)
case "${g3c4_mode}" in
  normal) g3c4_cflags+=(-O2) ;;
  sanitize) g3c4_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown G3-C4-N build mode: ${g3c4_mode}" ;;
esac
"${g3c4_cc}" "${g3c4_cflags[@]}" \
  -MMD -MF "${g3c4_temporary_dir}/g3c4_primitives_provider.d" -MT g3c4_primitives_provider.o \
  -c "${PROJECT_ROOT}/native/g3c4_primitives_provider.c" \
  -o "${g3c4_temporary_dir}/g3c4_primitives_provider.o"
ar rcsD "${g3c4_temporary_dir}/libeshkol_transformer_g3c4.a" \
  "${g3c4_temporary_dir}/g3c4_primitives_provider.o"
mkdir -p "${g3c4_artifact_dir}"
mv -f "${g3c4_temporary_dir}/g3c4_primitives_provider.o" "${g3c4_artifact_dir}/"
mv -f "${g3c4_temporary_dir}/g3c4_primitives_provider.d" "${g3c4_artifact_dir}/"
mv -f "${g3c4_temporary_dir}/libeshkol_transformer_g3c4.a" "${g3c4_artifact_dir}/"
printf 'built G3-C4-N carrier-neutral provider: %s\n' \
  "${g3c4_artifact_dir}/libeshkol_transformer_g3c4.a"
