#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${root}/build/mod5_moe"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
        -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math
        -fstack-protector-all -fPIC -I "${root}/include" -I "${root}/native")

"${CC}" "${cflags[@]}" -c "${root}/native/mod5_moe_provider.c" \
  -o "${scratch}/mod5_moe_provider.o"
ar rcsD "${scratch}/libeshkol_transformer_mod5_moe.a" \
  "${scratch}/mod5_moe_provider.o"

"${CC}" "${cflags[@]}" -c "${root}/native/mod5_moe_eshkol_bridge.c" \
  -o "${scratch}/mod5_moe_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_mod5_moe_bridge.a" \
  "${scratch}/mod5_moe_eshkol_bridge.o"

# Audit provider exports & undefined imports (zero allocator imports).
provider_exports="$(nm -g --defined-only "${scratch}/mod5_moe_provider.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_provider_exports="$(printf 'T %s\n' \
  et_mod5_moe_backward_f32 et_mod5_moe_forward_f32 et_mod5_moe_kernel_provider_v1)"
if [[ "${provider_exports}" != "${expected_provider_exports}" ]]; then
  printf 'unexpected MOD5 MoE provider exports:\n%s\n' "${provider_exports}" >&2
  exit 1
fi

provider_imports="$(nm -u "${scratch}/mod5_moe_provider.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    et_kernel_error_clear|exp|fegetenv|fegetround|fesetenv|memcpy|memset|\
    snprintf|strcmp) ;;
    *)
      printf 'unexpected MOD5 MoE provider dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${provider_imports}"

# Audit bridge exports & undefined imports (zero allocator imports).
bridge_exports="$(nm -g --defined-only "${scratch}/mod5_moe_eshkol_bridge.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_bridge_exports="$(printf 'T et_mod5_moe_aot_%s\n' \
  backward close forward load_f32 load_i64 open store_f32 store_i64)"
if [[ "${bridge_exports}" != "${expected_bridge_exports}" ]]; then
  printf 'unexpected MOD5 MoE bridge exports:\n%s\n' "${bridge_exports}" >&2
  exit 1
fi

bridge_imports="$(nm -u "${scratch}/mod5_moe_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|et_kernel_runtime_destroy|\
    et_kernel_runtime_discover|et_kernel_runtime_dispatch|\
    et_mod5_moe_kernel_provider_v1|memcpy|memset|strcmp) ;;
    *)
      printf 'unexpected MOD5 MoE bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f "${scratch}/libeshkol_transformer_mod5_moe.a" "${artifact}/"
mv -f "${scratch}/libeshkol_transformer_mod5_moe_bridge.a" "${artifact}/"
mv -f "${scratch}/mod5_moe_eshkol_bridge.o" "${artifact}/"
printf 'MOD5 MoE archives and symbol audit: PASS (%s/libeshkol_transformer_mod5_moe.a)\n' "${artifact}"
