#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
g3s_artifact_dir="${1:-$(project_build_dir)/g3s}"
g3s_mode="${2:-normal}"
g3s_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
mkdir -p "$(dirname -- "${g3s_artifact_dir}")"
g3s_temporary_dir="$(mktemp -d "$(dirname -- "${g3s_artifact_dir}")/.g3s-build.XXXXXX")"
trap 'rm -rf -- "${g3s_temporary_dir}"' EXIT
g3s_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fstack-usage -Wvla -Walloca -fPIC -I "${PROJECT_ROOT}/include"
)
case "${g3s_mode}" in
  normal) g3s_cflags+=(-O2) ;;
  sanitize) g3s_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown G3S build mode: ${g3s_mode}" ;;
esac
"${g3s_cc}" "${g3s_cflags[@]}" \
  -MMD -MF "${g3s_temporary_dir}/g3s_sampling_provider.d" -MT g3s_sampling_provider.o \
  -c "${PROJECT_ROOT}/native/g3s_sampling_provider.c" \
  -o "${g3s_temporary_dir}/g3s_sampling_provider.o"
ar rcsD "${g3s_temporary_dir}/libeshkol_transformer_g3s.a" \
  "${g3s_temporary_dir}/g3s_sampling_provider.o"
mkdir -p "${g3s_artifact_dir}"
mv -f "${g3s_temporary_dir}/g3s_sampling_provider.su" "${g3s_artifact_dir}/"
mv -f "${g3s_temporary_dir}/g3s_sampling_provider.o" "${g3s_artifact_dir}/"
mv -f "${g3s_temporary_dir}/g3s_sampling_provider.d" "${g3s_artifact_dir}/"
mv -f "${g3s_temporary_dir}/libeshkol_transformer_g3s.a" "${g3s_artifact_dir}/"
printf 'built G3S carrier-neutral provider: %s\n' \
  "${g3s_artifact_dir}/libeshkol_transformer_g3s.a"
