#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
cc="${CC}"
cxx="${CXX}"
artifact="${1:-${root}/build/wave4_composed_aot}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)

for trial in first second; do
  mkdir "${scratch}/${trial}"
  "${cc}" -std=c11 "${warnings[@]}" -O2 -ffp-contract=off \
    -fexcess-precision=standard -fno-fast-math -fstack-protector-all -fPIC \
    -I "${root}/include" -I "${root}/native" \
    -c "${root}/native/wave4_composed_eshkol_bridge.c" \
    -o "${scratch}/${trial}/wave4_composed_eshkol_bridge.o"
  ar rcsD "${scratch}/${trial}/libeshkol_transformer_wave4_composed_aot.a" \
    "${scratch}/${trial}/wave4_composed_eshkol_bridge.o"
done

archive="${scratch}/first/libeshkol_transformer_wave4_composed_aot.a"
cmp "${archive}" "${scratch}/second/libeshkol_transformer_wave4_composed_aot.a"

if [[ "$(ar t "${archive}")" != 'wave4_composed_eshkol_bridge.o' ]]; then
  printf 'unexpected Wave 4/5 composed AOT archive members\n' >&2
  exit 1
fi

exports="$(nm -g --defined-only "${archive}" |
  awk 'NF > 1 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected_exports="$(printf 'T et_wave4_composed_aot_%s\n' \
  activation_count bootstrap create destroy eval_contaminated_documents \
  eval_mean_nll eval_perplexity evaluate get_master last_local_overflow \
  last_loss last_status live_activation_peak_bytes manifest_digest_byte \
  manifest_digest_hex master_bits narrow_parameter_bytes param_digest_byte \
  param_digest_hex parameter_count planned_activation_peak_bytes \
  planned_workspace_bytes read_byte scaler_clean_steps scaler_scale \
  set_activation_limit set_input set_master set_overflow_inject set_target \
  step verify_manifest workspace_bytes)"
if [[ "${exports}" != "${expected_exports}" ]]; then
  printf 'unexpected Wave 4/5 composed AOT exports:\n%s\n' "${exports}" >&2
  exit 1
fi

imports="$(nm -u "${archive}" | awk 'NF > 1 {print $NF}' | LC_ALL=C sort -u)"
required_et="$(printf '%s\n' \
  et_ac4_run_v1 \
  et_dist5_comm_create \
  et_dist5_comm_destroy \
  et_eval5_manifest_read_file \
  et_mp4_loss_scaler_init \
  et_wave4_composed_bootstrap_v1 \
  et_wave4_composed_evaluate_v1 \
  et_wave4_composed_plan_v1 \
  et_wave4_composed_step_v1)"
actual_et="$(awk '/^et_/' <<< "${imports}")"
if [[ "${actual_et}" != "${required_et}" ]]; then
  printf 'unexpected Wave 4/5 composed AOT foundation imports:\n%s\n' "${actual_et}" >&2
  exit 1
fi

while IFS= read -r symbol; do
  case "${symbol}" in
    ''|et_*|_GLOBAL_OFFSET_TABLE_|__errno_location|__stack_chk_fail|\
    accept|bcmp|bind|calloc|clock_gettime|close|connect|fmaxf|free|\
    listen|memcmp|memcpy|memset|nanosleep|poll|recv|send|snprintf|\
    socket|strlen|unlink) ;;
    *)
      printf 'unexpected Wave 4/5 composed AOT dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

"${cxx}" -x c++ -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -fsyntax-only - <<'CPP'
#include "wave4_composed_eshkol_bridge.h"
static_assert(ET_WAVE4_COMPOSED_AOT_OUT_OF_MEMORY == 400);
static_assert(ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES == 65);
CPP

digest="$(sha256sum "${archive}")"
mv -f "${archive}" "${artifact}/"
printf 'Wave 4/5 composed AOT C11/C++17 build and symbol audit: PASS (SHA-256 %s)\n' \
  "${digest%% *}"
