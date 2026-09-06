#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

l2_artifact_dir="${1:-$(project_build_dir)/l2}"
l2_mode="${2:-normal}"
l2_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
l2_cc="$(tsv_value "${l2_provenance}" cc_path)"
l2_parent="$(dirname -- "${l2_artifact_dir}")"
mkdir -p "${l2_parent}"
l2_temporary="$(mktemp -d "${l2_parent}/.l2-build.XXXXXX")"
trap 'rm -rf -- "${l2_temporary}"' EXIT

l2_cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -I "${PROJECT_ROOT}/include")
case "${l2_mode}" in
  normal) ;;
  sanitize) l2_cflags+=(-fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown L2 build mode: ${l2_mode}" ;;
esac

"${l2_cc}" "${l2_cflags[@]}" -c \
  "${PROJECT_ROOT}/native/indexed_cross_entropy.c" \
  -o "${l2_temporary}/indexed_cross_entropy.o"
ar rcsD "${l2_temporary}/libeshkol_transformer_l2.a" \
  "${l2_temporary}/indexed_cross_entropy.o"
mkdir -p "${l2_artifact_dir}"
mv -f "${l2_temporary}/indexed_cross_entropy.o" "${l2_artifact_dir}/"
mv -f "${l2_temporary}/libeshkol_transformer_l2.a" "${l2_artifact_dir}/"
printf 'built L2 indexed cross-entropy: %s\n' \
  "${l2_artifact_dir}/libeshkol_transformer_l2.a"
