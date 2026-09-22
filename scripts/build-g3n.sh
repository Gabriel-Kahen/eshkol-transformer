#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
g3n_artifact_dir="${1:-$(project_build_dir)/g3n}"
g3n_mode="${2:-normal}"
g3n_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
mkdir -p "$(dirname -- "${g3n_artifact_dir}")"
g3n_temporary_dir="$(mktemp -d "$(dirname -- "${g3n_artifact_dir}")/.g3n-build.XXXXXX")"
trap 'rm -rf -- "${g3n_temporary_dir}"' EXIT
g3n_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC -I "${PROJECT_ROOT}/include"
)
case "${g3n_mode}" in
  normal) g3n_cflags+=(-O2) ;;
  sanitize) g3n_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown G3N build mode: ${g3n_mode}" ;;
esac
"${g3n_cc}" "${g3n_cflags[@]}" \
  -MMD -MF "${g3n_temporary_dir}/g3n_primitives_provider.d" -MT g3n_primitives_provider.o \
  -c "${PROJECT_ROOT}/native/g3n_primitives_provider.c" \
  -o "${g3n_temporary_dir}/g3n_primitives_provider.o"
ar rcsD "${g3n_temporary_dir}/libeshkol_transformer_g3n.a" \
  "${g3n_temporary_dir}/g3n_primitives_provider.o"
mkdir -p "${g3n_artifact_dir}"
mv -f "${g3n_temporary_dir}/g3n_primitives_provider.o" "${g3n_artifact_dir}/"
mv -f "${g3n_temporary_dir}/g3n_primitives_provider.d" "${g3n_artifact_dir}/"
mv -f "${g3n_temporary_dir}/libeshkol_transformer_g3n.a" "${g3n_artifact_dir}/"
printf 'built G3N carrier-neutral provider: %s\n' \
  "${g3n_artifact_dir}/libeshkol_transformer_g3n.a"
