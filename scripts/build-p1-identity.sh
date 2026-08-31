#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar

output_root="${1:-$(project_build_dir)/p1}"
build_mode="${2:-normal}"
product="${3:-public}"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
parent="$(dirname -- "${output_root}")"
temporary="$(mktemp -d "${parent}/.p1-identity.XXXXXX")"
trap 'rm -rf -- "${temporary}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common
  -I "${PROJECT_ROOT}/native"
)
case "${build_mode}" in
  normal) ;;
  sanitize)
    cflags+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
    ;;
  *) die "unknown P1 identity build mode: ${build_mode}" ;;
esac

case "${product}" in
  public|trusted|all) ;;
  *) die "unknown P1 identity product: ${product}" ;;
esac

if [[ "${product}" == public || "${product}" == all ]]; then
  mkdir -p "${temporary}/public"
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/p1_identity.c" \
    -o "${temporary}/public/p1_identity.o"
  ar rcsD "${temporary}/public/libeshkol_transformer_p1_identity.a" \
    "${temporary}/public/p1_identity.o"
fi

if [[ "${product}" == trusted || "${product}" == all ]]; then
  mkdir -p "${temporary}/trusted"
  "${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
    -c "${PROJECT_ROOT}/native/p1_identity.c" \
    -o "${temporary}/trusted/p1_identity.o"
  ar rcsD "${temporary}/trusted/libeshkol_transformer_p1_identity.a" \
    "${temporary}/trusted/p1_identity.o"
fi

mkdir -p "${output_root}"
if [[ "${product}" == public || "${product}" == all ]]; then
  rm -rf -- "${output_root}/public"
  mv "${temporary}/public" "${output_root}/public"
fi
if [[ "${product}" == public ]]; then
  rm -rf -- "${output_root}/trusted"
fi
if [[ "${product}" == trusted || "${product}" == all ]]; then
  rm -rf -- "${output_root}/trusted"
  mv "${temporary}/trusted" "${output_root}/trusted"
fi
printf 'built P1 identity product %s: %s\n' "${product}" "${output_root}"
