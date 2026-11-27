#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile=width16
opt_in=(-DET_I2_PRIVATE_MOD5_WIDTH16_STORAGE_COPY)
private_symbols=(et_i2_private_mod5_width16_storage_provider_v1)
if [[ ${1:-} == --v256 ]]; then
  shift
  profile=width16-v256
  opt_in+=(-DET_I2_PRIVATE_MOD5_WIDTH16_V256_STORAGE_COPY)
  private_symbols+=(et_i2_private_mod5_width16_v256_storage_provider_v1)
fi
if (( $# > 1 )); then
  printf 'usage: %s [--v256] [evidence-directory]\n' "$0" >&2
  exit 2
fi
evidence="${1:-${root}/build/mod5-${profile}-storage-copy}"
cc="${CC:-clang}"
mkdir -p "${evidence}"
evidence="$(cd -- "${evidence}" && pwd)"
export LC_ALL=C

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -frounding-math -fPIC -fvisibility=hidden
       -fno-common -fstack-protector-all -I "${root}/include" -I "${root}/native")
storage="${root}/native/f32_tensor.c"
k1="${root}/native/kernel_abi.c"

roster="${root}/tests/mod5/mod5_e2e_c1_io5.esk"
head_shape='16 16'
if [[ ${profile} == width16-v256 ]]; then
  input="$(mktemp -d "${evidence}/v256-input.XXXXXX")"
  (
    cd -- "${root}"
    python3 -m tests.eval5.mod5_v256_fixture build "${input}/fixture" >/dev/null
    python3 -m tests.eval5.build_mod5_v256_witness "${input}/fixture" "${input}/model.esk"
  )
  roster="${input}/model.esk"
  head_shape='256 16'
fi
sed -nE '/^\(define roster$/,/^\(define tied-aliases /p' "${roster}" >"${evidence}/roster.txt"
grep -F '(("head" "weight") ('"${head_shape}"'))' "${evidence}/roster.txt" >/dev/null
grep -F '(("token_embedding" "weight") ('"${head_shape}"'))' "${evidence}/roster.txt" >/dev/null
sed -nE '/^\(define roster$/,/^\(define tied-aliases / s/.*\) \(([0-9]+( [0-9]+)*)\)\)+$/\1/p' \
  "${roster}" >"${evidence}/roster-shapes.txt"
printf '%s\n' '8 16' '16 16' '16 16' '8 16' '16 32' '32 16' '32 16' \
  '16' '16' "${head_shape}" '16' "${head_shape}" | cmp - "${evidence}/roster-shapes.txt"

"${cc}" "${flags[@]}" -c "${storage}" -o "${evidence}/default.o"
"${cc}" "${flags[@]}" "${opt_in[@]}" -c "${storage}" -o "${evidence}/${profile}.o"
for artifact in default "${profile}"; do
  nm -g --defined-only --format=posix "${evidence}/${artifact}.o" |
    awk '{print $1}' | sort >"${evidence}/${artifact}-defined.txt"
  nm -u --format=posix "${evidence}/${artifact}.o" |
    awk '{print $1}' | sort -u >"${evidence}/${artifact}-undefined.txt"
done
cmp "${root}/native/f32_tensor_defined_symbols.txt" "${evidence}/default-defined.txt"
cmp "${root}/native/f32_tensor_undefined_symbols.txt" "${evidence}/default-undefined.txt"
cmp "${evidence}/default-undefined.txt" "${evidence}/${profile}-undefined.txt"
{ cat "${evidence}/default-defined.txt"; printf '%s\n' "${private_symbols[@]}"; } |
  sort | cmp - "${evidence}/${profile}-defined.txt"
readelf --wide --syms "${evidence}/${profile}.o" >"${evidence}/${profile}-readelf.txt"
for private_symbol in "${private_symbols[@]}"; do
  awk -v symbol="${private_symbol}" \
    '$5 == "GLOBAL" && $6 == "HIDDEN" && $8 == symbol {found = 1} END {exit !found}' \
    "${evidence}/${profile}-readelf.txt"
done

"${cc}" "${flags[@]}" "${root}/tests/k2/report_i2.c" "${k1}" \
  "${evidence}/${profile}.o" -lm -o "${evidence}/report-default"
"${evidence}/report-default" >"${evidence}/default.json"

run_mode() {
  local mode="$1"
  local -a mode_flags=(-O2)
  if [[ "${mode}" == sanitized ]]; then
    mode_flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "${cc}" "${flags[@]}" "${mode_flags[@]}" "${opt_in[@]}" \
    "${root}/tests/i2/test_mod5_width16_storage_copy.c" "${storage}" "${k1}" \
    -lm -o "${evidence}/copy-${mode}"
  "${cc}" "${flags[@]}" "${mode_flags[@]}" "${opt_in[@]}" -DET_K2_TESTING \
    "${root}/tests/k2/test_capabilities_native.c" \
    "${root}/native/k2_capabilities.c" "${storage}" "${k1}" \
    -lm -o "${evidence}/k2-${mode}"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "${evidence}/copy-${mode}" >"${evidence}/${profile}-${mode}.json"
  if [[ ${profile} == width16-v256 ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      "${evidence}/copy-${mode}" --report-v16 >"${evidence}/width16-${mode}.json"
  fi
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "${evidence}/k2-${mode}" >"${evidence}/k2-${mode}.stdout"
}

run_mode normal
run_mode sanitized
if [[ ${profile} == width16-v256 ]]; then
  "${evidence}/copy-normal" >"${evidence}/${profile}-repeat.json"
  cmp "${evidence}/${profile}-normal.json" "${evidence}/${profile}-repeat.json"
  cmp "${evidence}/${profile}-normal.json" "${evidence}/${profile}-sanitized.json"
fi
cmp "${evidence}/width16-normal.json" "${evidence}/width16-sanitized.json"
cmp "${evidence}/k2-normal.stdout" "${evidence}/k2-sanitized.stdout"
grep -Fx 'K2 native capability checks passed' "${evidence}/k2-normal.stdout" >/dev/null
(
  cd -- "${evidence}"
  sha256sum --check <<'END'
50c7078af4d3e495c4d668b6b8aaab29024cbab6a775fa6e98c9e9d10124389e  default.json
c53dbd9b29b17ced5ef8a84ece7f96bb5c4d6b3cb269106366265a54e6008b40  width16-normal.json
END
  if [[ ${profile} == width16-v256 ]]; then
    sha256sum --check <<'END'
50e09d8e0bf757492e1a0c10aacaf58ba124d18f6e48b65cf3f21f41dd288ab6  width16-v256-normal.json
END
  fi
)
printf 'MOD5 %s I2 copy and default K2 rejection passed. Evidence: %s\n' "${profile}" "${evidence}"
