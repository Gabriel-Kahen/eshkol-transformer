#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evidence="${1:-${root}/build/mod5-width16-storage-copy}"
cc="${CC:-clang}"
mkdir -p "${evidence}"
export LC_ALL=C

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -frounding-math -fPIC -fvisibility=hidden
       -fno-common -fstack-protector-all -I "${root}/include" -I "${root}/native")
opt_in=-DET_I2_PRIVATE_MOD5_WIDTH16_STORAGE_COPY
storage="${root}/native/f32_tensor.c"
k1="${root}/native/kernel_abi.c"

# Match the twelve logical C1 rows in the merged live MOD5-C witness.
roster="${root}/tests/mod5/mod5_e2e_c1_io5.esk"
sed -nE '/^\(define roster$/,/^\(define tied-aliases / s/.*\) \(([0-9]+( [0-9]+)*)\)\)+$/\1/p' \
  "${roster}" >"${evidence}/roster-shapes.txt"
printf '%s\n' '8 16' '16 16' '16 16' '8 16' '16 32' '32 16' '32 16' \
  '16' '16' '16 16' '16' '16 16' | cmp - "${evidence}/roster-shapes.txt"

"${cc}" "${flags[@]}" -c "${storage}" -o "${evidence}/default.o"
"${cc}" "${flags[@]}" "${opt_in}" -c "${storage}" -o "${evidence}/width16.o"
for profile in default width16; do
  nm -g --defined-only --format=posix "${evidence}/${profile}.o" |
    awk '{print $1}' | sort >"${evidence}/${profile}-defined.txt"
  nm -u --format=posix "${evidence}/${profile}.o" |
    awk '{print $1}' | sort -u >"${evidence}/${profile}-undefined.txt"
done
cmp "${root}/native/f32_tensor_defined_symbols.txt" "${evidence}/default-defined.txt"
cmp "${root}/native/f32_tensor_undefined_symbols.txt" "${evidence}/default-undefined.txt"
cmp "${evidence}/default-undefined.txt" "${evidence}/width16-undefined.txt"
private_symbol=et_i2_private_mod5_width16_storage_provider_v1
{ cat "${evidence}/default-defined.txt"; printf '%s\n' "${private_symbol}"; } |
  sort | cmp - "${evidence}/width16-defined.txt"
readelf --wide --syms "${evidence}/width16.o" |
  awk -v symbol="${private_symbol}" \
    '$5 == "GLOBAL" && $6 == "HIDDEN" && $8 == symbol {found = 1} END {exit !found}'

"${cc}" "${flags[@]}" "${root}/tests/k2/report_i2.c" "${k1}" \
  "${evidence}/width16.o" -lm -o "${evidence}/report-default"
"${evidence}/report-default" >"${evidence}/default.json"

run_mode() {
  local mode="$1"
  local -a mode_flags=(-O2)
  if [[ "${mode}" == sanitized ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${cc}" "${flags[@]}" "${mode_flags[@]}" "${opt_in}" \
    "${root}/tests/i2/test_mod5_width16_storage_copy.c" "${storage}" "${k1}" \
    -lm -o "${evidence}/copy-${mode}"
  "${cc}" "${flags[@]}" "${mode_flags[@]}" "${opt_in}" -DET_K2_TESTING \
    "${root}/tests/k2/test_capabilities_native.c" \
    "${root}/native/k2_capabilities.c" "${storage}" "${k1}" \
    -lm -o "${evidence}/k2-${mode}"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "${evidence}/copy-${mode}" >"${evidence}/width16-${mode}.json"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "${evidence}/k2-${mode}" >"${evidence}/k2-${mode}.stdout"
}

run_mode normal
run_mode sanitized
cmp "${evidence}/width16-normal.json" "${evidence}/width16-sanitized.json"
cmp "${evidence}/k2-normal.stdout" "${evidence}/k2-sanitized.stdout"
grep -Fx 'K2 native capability checks passed' "${evidence}/k2-normal.stdout" >/dev/null
(
  cd -- "${evidence}"
  sha256sum --check <<'END'
50c7078af4d3e495c4d668b6b8aaab29024cbab6a775fa6e98c9e9d10124389e  default.json
c53dbd9b29b17ced5ef8a84ece7f96bb5c4d6b3cb269106366265a54e6008b40  width16-normal.json
END
)
printf 'MOD5 width-16 I2 copy and default K2 rejection passed. Evidence: %s\n' "${evidence}"
