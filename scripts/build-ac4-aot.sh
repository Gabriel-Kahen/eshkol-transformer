#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
artifact="${1:-${root}/build/ac4_aot}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
"${cc}" -std=c11 "${warnings[@]}" -O2 -ffp-contract=off \
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all -fPIC \
  -I "${root}/include" -I "${root}/native" \
  -c "${root}/native/ac4_eshkol_bridge.c" -o "${scratch}/ac4_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_ac4_aot.a" "${scratch}/ac4_eshkol_bridge.o"
archive="${scratch}/libeshkol_transformer_ac4_aot.a"

if [[ "$(ar t "${archive}")" != 'ac4_eshkol_bridge.o' ]]; then
  printf 'unexpected AC4 AOT archive members\n' >&2
  exit 1
fi

exports="$(nm -g --defined-only "${archive}" |
  awk 'NF > 1 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected_exports="$(printf 'T et_ac4_aot_%s\n' \
  activation_floats create d_input_bits d_parameter_bits destroy \
  get_d_input get_d_parameter get_loss get_output get_parameter \
  gradient_staging_bytes loss_bits_hi loss_bits_lo output_bits \
  parameter_bits parameter_floats parameter_floats_per_layer \
  peak_activation_bytes planned_gradient_staging_bytes planned_peak_bytes \
  planned_rematerialized_layers planned_retained_checkpoints \
  planned_retained_tapes planned_workspace_bytes recomputation_count \
  retained_checkpoints retained_tapes run set_activation_limit \
  set_input set_parameter set_target sgd_step workspace_bytes)"
if [[ "${exports}" != "${expected_exports}" ]]; then
  printf 'unexpected AC4 AOT exports:\n%s\n' "${exports}" >&2
  exit 1
fi

imports="$(nm -u "${archive}" | awk 'NF > 1 {print $NF}')"
uses_plan=0
uses_run=0
while IFS= read -r symbol; do
  case "${symbol}" in
    et_ac4_plan_v1) uses_plan=1 ;;
    et_ac4_run_v1) uses_run=1 ;;
    ''|_GLOBAL_OFFSET_TABLE_|__stack_chk_fail|calloc|free|memcpy|memset) ;;
    *)
      printf 'unexpected AC4 AOT dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"
if [[ "${uses_plan}" != 1 || "${uses_run}" != 1 ]]; then
  printf 'missing AC4 AOT dependency on AC4 ABI\n' >&2
  exit 1
fi

"${cxx}" -x c++ -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -fsyntax-only - <<'CPP'
#include "ac4_eshkol_bridge.h"
static_assert(ET_AC4_AOT_OUT_OF_MEMORY == 100);
CPP

mv -f "${archive}" "${artifact}/"
printf 'AC4 AOT C11/C++17 build and symbol audit: PASS (%s)\n' \
  "${artifact}/libeshkol_transformer_ac4_aot.a"
