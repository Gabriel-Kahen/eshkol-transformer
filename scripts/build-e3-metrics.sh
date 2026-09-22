#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
require_command ar
e3_metrics_artifact_dir="${1:-$(project_build_dir)/e3-metrics}"
e3_metrics_mode="${2:-normal}"
e3_metrics_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
mkdir -p "$(dirname -- "${e3_metrics_artifact_dir}")"
e3_metrics_temporary_dir="$(mktemp -d "$(dirname -- "${e3_metrics_artifact_dir}")/.e3_metrics-build.XXXXXX")"
trap 'rm -rf -- "${e3_metrics_temporary_dir}"' EXIT
e3_metrics_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math -frounding-math
  -fstack-protector-all -fPIC -I "${PROJECT_ROOT}/include"
)
case "${e3_metrics_mode}" in
  normal) e3_metrics_cflags+=(-O2) ;;
  sanitize) e3_metrics_cflags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown E3 metrics build mode: ${e3_metrics_mode}" ;;
esac
"${e3_metrics_cc}" "${e3_metrics_cflags[@]}" \
  -MMD -MF "${e3_metrics_temporary_dir}/e3_evaluation_metrics_provider.d" -MT e3_evaluation_metrics_provider.o \
  -c "${PROJECT_ROOT}/native/e3_evaluation_metrics_provider.c" \
  -o "${e3_metrics_temporary_dir}/e3_evaluation_metrics_provider.o"
ar rcsD "${e3_metrics_temporary_dir}/libeshkol_transformer_e3_metrics.a" \
  "${e3_metrics_temporary_dir}/e3_evaluation_metrics_provider.o"
mkdir -p "${e3_metrics_artifact_dir}"
mv -f "${e3_metrics_temporary_dir}/e3_evaluation_metrics_provider.o" "${e3_metrics_artifact_dir}/"
mv -f "${e3_metrics_temporary_dir}/e3_evaluation_metrics_provider.d" "${e3_metrics_artifact_dir}/"
mv -f "${e3_metrics_temporary_dir}/libeshkol_transformer_e3_metrics.a" "${e3_metrics_artifact_dir}/"
printf 'built E3 metrics carrier-neutral provider: %s\n' \
  "${e3_metrics_artifact_dir}/libeshkol_transformer_e3_metrics.a"
