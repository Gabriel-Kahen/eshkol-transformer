#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${1:-${root}/build/flash_attn2}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
cflags=(-std=c11 "${warnings[@]}" -O2 -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math
        -fstack-protector-all -fPIC -I "${root}/include" -I "${root}/native")

"${CC}" "${cflags[@]}" -c "${root}/native/flash_attn2.c" \
  -o "${scratch}/flash_attn2.o"
ar rcsD "${scratch}/libeshkol_transformer_flash_attn2.a" \
  "${scratch}/flash_attn2.o"

"${CC}" "${cflags[@]}" -c "${root}/native/flash_attn2_eshkol_bridge.c" \
  -o "${scratch}/flash_attn2_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_flash_attn2_aot.a" \
  "${scratch}/flash_attn2_eshkol_bridge.o"

# Audit core FlashAttention-2 exports & undefined imports (strictly ZERO allocator imports).
core_exports="$(nm -g --defined-only "${scratch}/flash_attn2.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_core_exports="$(printf 'T %s\n' \
  et_fa2_backward_v1 \
  et_fa2_forward_v1 \
  et_fa2_tile_scratch_bytes_v1 \
  et_fa2_validate_config_v1)"
if [[ "${core_exports}" != "${expected_core_exports}" ]]; then
  printf 'unexpected FlashAttention-2 core exports:\n%s\n' "${core_exports}" >&2
  exit 1
fi

core_imports="$(nm -u "${scratch}/flash_attn2.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    exp|fegetenv|fegetround|fesetenv|log|memcpy|memset|sqrt|tanh) ;;
    *)
      printf 'unexpected FlashAttention-2 core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit Eshkol AOT bridge exports & undefined imports (strictly ZERO allocator imports).
bridge_exports="$(nm -g --defined-only "${scratch}/flash_attn2_eshkol_bridge.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_bridge_exports="$(printf 'T et_fa2_aot_%s\n' \
  backward check_bad_fpenv_forward config_bytes forward load_f32 load_f64 \
  set_config stats_bytes stats_evaluated_tiles stats_skipped_tiles \
  stats_tile_scratch_bytes stats_total_tiles store_f32 store_f64 \
  tile_scratch_bytes)"
if [[ "${bridge_exports}" != "${expected_bridge_exports}" ]]; then
  printf 'unexpected FlashAttention-2 AOT bridge exports:\n%s\n' "${bridge_exports}" >&2
  exit 1
fi

bridge_imports="$(nm -u "${scratch}/flash_attn2_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    et_fa2_backward_v1|et_fa2_forward_v1|et_fa2_tile_scratch_bytes_v1|\
    fegetenv|fesetenv|fesetround|memcpy|memset) ;;
    *)
      printf 'unexpected FlashAttention-2 AOT bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

# C++17 header syntax verification.
"${CXX}" -x c++ -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -fsyntax-only - <<'CPP'
#include "eshkol_transformer/flash_attn2_abi.h"
#include "flash_attn2_eshkol_bridge.h"
static_assert(ET_FA2_ABI_MAJOR == 1u);
static_assert(ET_FA2_OK == 0);
CPP

mv -f "${scratch}/libeshkol_transformer_flash_attn2.a" "${artifact}/"
mv -f "${scratch}/libeshkol_transformer_flash_attn2_aot.a" "${artifact}/"
printf 'FLASH_ATTN2 deterministic archives and symbol audit: PASS (%s/libeshkol_transformer_flash_attn2.a)\n' "${artifact}"
