#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${1:-${root}/build/mamba_ssm}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
cflags=(-std=c11 "${warnings[@]}" -O2 -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math
        -fstack-protector-all -fPIC -I "${root}/include" -I "${root}/native")

"${CC}" "${cflags[@]}" -c "${root}/native/mamba_ssm.c" \
  -o "${scratch}/mamba_ssm.o"
ar rcsD "${scratch}/libeshkol_transformer_mamba_ssm.a" \
  "${scratch}/mamba_ssm.o"

"${CC}" "${cflags[@]}" -c "${root}/native/mamba_ssm_eshkol_bridge.c" \
  -o "${scratch}/mamba_ssm_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_mamba_ssm_aot.a" \
  "${scratch}/mamba_ssm_eshkol_bridge.o"

# Audit core Mamba S6 exports & undefined imports (strictly zero allocator imports).
core_exports="$(nm -g --defined-only "${scratch}/mamba_ssm.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_core_exports="$(printf 'T %s\n' \
  et_mamba_cache_conv_state_bytes_v1 \
  et_mamba_cache_init_v1 \
  et_mamba_cache_reset_v1 \
  et_mamba_cache_ssm_state_bytes_v1 \
  et_mamba_cache_step_v1 \
  et_mamba_cache_total_state_bytes_v1 \
  et_mamba_causal_conv1d_silu_backward_v1 \
  et_mamba_causal_conv1d_silu_forward_v1 \
  et_mamba_selective_scan_backward_v1 \
  et_mamba_selective_scan_forward_v1 \
  et_mamba_validate_config_v1)"
if [[ "${core_exports}" != "${expected_core_exports}" ]]; then
  printf 'unexpected Mamba S6 core exports:\n%s\n' "${core_exports}" >&2
  exit 1
fi

core_imports="$(nm -u "${scratch}/mamba_ssm.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    exp|fegetenv|fegetround|fesetenv|log1p|memcpy|memset) ;;
    *)
      printf 'unexpected Mamba S6 core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit Eshkol AOT bridge exports & undefined imports (strictly zero allocator imports).
bridge_exports="$(nm -g --defined-only "${scratch}/mamba_ssm_eshkol_bridge.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_bridge_exports="$(printf 'T et_mamba_aot_%s\n' \
  cache_bytes cache_init cache_reset cache_step cache_step_count config_bytes \
  conv1d_silu_backward conv1d_silu_forward conv_state_bytes load_f32 null_ptr \
  selective_scan_backward selective_scan_forward set_config ssm_state_bytes \
  store_f32 total_state_bytes)"
if [[ "${bridge_exports}" != "${expected_bridge_exports}" ]]; then
  printf 'unexpected Mamba S6 AOT bridge exports:\n%s\n' "${bridge_exports}" >&2
  exit 1
fi

bridge_imports="$(nm -u "${scratch}/mamba_ssm_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    et_mamba_cache_conv_state_bytes_v1|et_mamba_cache_init_v1|\
    et_mamba_cache_reset_v1|et_mamba_cache_ssm_state_bytes_v1|\
    et_mamba_cache_step_v1|et_mamba_cache_total_state_bytes_v1|\
    et_mamba_causal_conv1d_silu_backward_v1|\
    et_mamba_causal_conv1d_silu_forward_v1|\
    et_mamba_selective_scan_backward_v1|\
    et_mamba_selective_scan_forward_v1|\
    memcpy|memset) ;;
    *)
      printf 'unexpected Mamba S6 AOT bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

# C++17 header syntax verification.
"${CXX}" -x c++ -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -fsyntax-only - <<'CPP'
#include "eshkol_transformer/mamba_ssm_abi.h"
#include "mamba_ssm_eshkol_bridge.h"
static_assert(ET_MAMBA_SSM_ABI_MAJOR == 1u);
static_assert(ET_MAMBA_OK == 0);
CPP

mv -f "${scratch}/libeshkol_transformer_mamba_ssm.a" "${artifact}/"
mv -f "${scratch}/libeshkol_transformer_mamba_ssm_aot.a" "${artifact}/"
printf 'MAMBA_SSM deterministic archives and symbol audit: PASS (%s/libeshkol_transformer_mamba_ssm.a)\n' "${artifact}"
