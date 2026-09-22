#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
m3cg_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
m3cg_dir="$(project_build_dir)/m3cg/native"
mkdir -p "${m3cg_dir}"
m3cg_flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -frounding-math -fstack-protector-all
  -DET_F32_TENSOR_TESTING -DET_M3_CALL_TESTING
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
for m3cg_mode in normal sanitize; do
  m3cg_extra=(-O2)
  if [[ "${m3cg_mode}" == sanitize ]]; then
    m3cg_extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${m3cg_cc}" "${m3cg_flags[@]}" "${m3cg_extra[@]}" \
    "${PROJECT_ROOT}/tests/m3cg/test_pins.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -lm -o "${m3cg_dir}/pins-${m3cg_mode}"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${m3cg_dir}/pins-${m3cg_mode}" >"${m3cg_dir}/${m3cg_mode}.stdout" \
    2>"${m3cg_dir}/${m3cg_mode}.stderr"
  cat "${m3cg_dir}/${m3cg_mode}.stdout"
done
