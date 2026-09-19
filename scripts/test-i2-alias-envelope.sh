#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
k1="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
[[ -s "${k1}" ]] || die "I2 alias envelope tests require canonical K1 archive"
output="$(project_build_dir)/i2-alias-envelope"
mkdir -p "${output}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -frounding-math -I "${PROJECT_ROOT}/include"
       -I "${PROJECT_ROOT}/native")
for variant in normal sanitized; do
  extra=()
  if [[ "${variant}" == sanitized ]]; then extra=(-fsanitize=address,undefined -fno-omit-frame-pointer); fi
  "${cc}" "${flags[@]}" "${extra[@]}" "${PROJECT_ROOT}/tests/i2/test_alias_envelope.c" \
    "${k1}" -o "${output}/${variant}"
  for mode in coverage product endpoint uninitialized; do
    ASAN_OPTIONS="detect_leaks=${I2_ASAN_DETECT_LEAKS:-1}:halt_on_error=1" UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 60s "${output}/${variant}" "${mode}" \
      >"${output}/${variant}-${mode}.stdout" 2>"${output}/${variant}-${mode}.stderr"
    cat "${output}/${variant}-${mode}.stdout"
  done
done
