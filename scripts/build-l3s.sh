#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
l3s_artifact_dir="${1:-$(project_build_dir)/l3s}"
l3s_mode="${2:-normal}"
l3s_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
mkdir -p "$(dirname -- "${l3s_artifact_dir}")"
l3s_temporary_dir="$(mktemp -d "$(dirname -- "${l3s_artifact_dir}")/.l3s-build.XXXXXX")"
trap 'rm -rf -- "${l3s_temporary_dir}"' EXIT
l3s_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math -frounding-math
  -fstack-protector-all -fPIC -I "${PROJECT_ROOT}/include"
)
case "${l3s_mode}" in
  normal) l3s_cflags+=(-O2) ;;
  sanitize) l3s_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown L3S build mode: ${l3s_mode}" ;;
esac
"${l3s_cc}" "${l3s_cflags[@]}" \
  -MMD -MF "${l3s_temporary_dir}/l3s_masked_objective_provider.d" -MT l3s_masked_objective_provider.o \
  -c "${PROJECT_ROOT}/native/l3s_masked_objective_provider.c" \
  -o "${l3s_temporary_dir}/l3s_masked_objective_provider.o"
ar rcsD "${l3s_temporary_dir}/libeshkol_transformer_l3s.a" \
  "${l3s_temporary_dir}/l3s_masked_objective_provider.o"
mkdir -p "${l3s_artifact_dir}"
mv -f "${l3s_temporary_dir}/l3s_masked_objective_provider.o" "${l3s_artifact_dir}/"
mv -f "${l3s_temporary_dir}/l3s_masked_objective_provider.d" "${l3s_artifact_dir}/"
mv -f "${l3s_temporary_dir}/libeshkol_transformer_l3s.a" "${l3s_artifact_dir}/"
printf 'built L3S carrier-neutral provider: %s\n' \
  "${l3s_artifact_dir}/libeshkol_transformer_l3s.a"
