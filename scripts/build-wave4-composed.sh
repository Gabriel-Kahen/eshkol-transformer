#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
artifact=build/wave4-composed
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
warnings=(-Wall -Wextra -Werror -Wpedantic -I include)

"${cxx}" -x c++ -std=c++17 "${warnings[@]}" -fsyntax-only - <<'CPP'
#include "eshkol_transformer/wave4_composed_trainer_abi.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_wave4_composed_config_v1>);
static_assert(std::is_standard_layout_v<et_wave4_composed_state_v1>);
static_assert(std::is_standard_layout_v<et_wave4_composed_result_v1>);
static_assert(std::is_standard_layout_v<et_wave4_composed_eval_v1>);
CPP

for trial in first second; do
  mkdir "${scratch}/${trial}"
  "${cc}" -std=c11 "${warnings[@]}" -O2 -ffp-contract=off -fno-fast-math \
    -fexcess-precision=standard -fstack-protector-all -fPIC \
    -c native/wave4_composed_trainer.c -o "${scratch}/${trial}/wave4_composed_trainer.o"
  ar rcsD "${scratch}/${trial}/libeshkol_wave4_composed.a" \
    "${scratch}/${trial}/wave4_composed_trainer.o"
done

archive="${scratch}/first/libeshkol_wave4_composed.a"
cmp "${archive}" "${scratch}/second/libeshkol_wave4_composed.a"
exports="$(nm -g --defined-only "${archive}" |
  awk 'NF == 3 {print $2, $3}' | LC_ALL=C sort)"
expected=(et_wave4_composed_bootstrap_v1 et_wave4_composed_evaluate_v1
          et_wave4_composed_plan_v1 et_wave4_composed_step_v1)
if [[ "${exports}" != "$(printf 'T %s\n' "${expected[@]}")" ]]; then
  printf 'unexpected composed trainer exports:\n%s\n' "${exports}" >&2
  exit 1
fi

imports="$(nm -u "${archive}" | awk '$1 == "U" {print $2}' | LC_ALL=C sort)"
required=(et_ac4_plan_v1 et_ac4_run_v1 et_dist5_allreduce et_dist5_broadcast
          et_dist5_sync_gradients_f32 et_eval5_audit_contamination
          et_eval5_manifest_format et_eval5_manifest_read_file
          et_eval5_manifest_write_atomic et_eval5_score_lm
          et_kernel_runtime_destroy et_kernel_runtime_discover et_kernel_runtime_dispatch
          et_mp4_kernel_provider_v1 et_mp4_loss_scaler_init et_mp4_loss_scaler_update)
if [[ "$(awk '/^et_/' <<< "${imports}")" != "$(printf '%s\n' "${required[@]}")" ]]; then
  printf 'unexpected composed trainer foundation imports:\n%s\n' "${imports}" >&2
  exit 1
fi
while IFS= read -r symbol; do
  case "${symbol}" in
    et_*|_GLOBAL_OFFSET_TABLE_|__stack_chk_fail|bcmp|memcmp|memcpy|memset|strcmp|strlen) ;;
    *) printf 'unexpected composed trainer import: %s\n' "${symbol}" >&2; exit 1 ;;
  esac
done <<< "${imports}"
digest="$(sha256sum "${archive}")"
mv -f "${archive}" "${artifact}/"
printf 'Wave 4/5 composed C11/C++17 build and symbol audit: PASS (SHA-256 %s)\n' "${digest%% *}"
