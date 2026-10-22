#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
require_command python3
require_command timeout
cpu4_dir="${1:-$(project_build_dir)/cpu4-primitives}"
cpu4_mode="${2:-normal}"
mkdir -p "${cpu4_dir}"
cpu4_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
cpu4_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${cpu4_dir}")
case "${cpu4_mode}" in
  normal) cpu4_flags+=(-O2) ;;
  sanitize) cpu4_flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) ;;
  *) die "unknown CPU4 primitive gate mode: ${cpu4_mode}" ;;
esac
for cpu4_provider in k1 a2 n2 n3k; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-${cpu4_provider}.sh" \
    "${cpu4_dir}/${cpu4_provider}" "${cpu4_mode}"
done
python3 "${PROJECT_ROOT}/tests/cpu4_primitives/generate_reference.py" "${cpu4_dir}/reference.h"
python3 "${PROJECT_ROOT}/tests/cpu4_primitives/generate_reference.py" "${cpu4_dir}/reference-repeat.h"
cmp "${cpu4_dir}/reference.h" "${cpu4_dir}/reference-repeat.h"
"${cpu4_cc}" "${cpu4_flags[@]}" "${PROJECT_ROOT}/tests/cpu4_primitives/test_primitives.c" \
  "${cpu4_dir}/n3k/libeshkol_transformer_n3k.a" \
  "${cpu4_dir}/n2/libeshkol_transformer_n2.a" \
  "${cpu4_dir}/a2/libeshkol_transformer_a2.a" \
  "${cpu4_dir}/k1/libeshkol_transformer_k1.a" -lm -ldl -o "${cpu4_dir}/test-cpu4-primitives"
timeout --foreground --signal=TERM --kill-after=5s 60s "${cpu4_dir}/test-cpu4-primitives"
