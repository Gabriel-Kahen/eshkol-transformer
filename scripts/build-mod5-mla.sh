#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${1:-${root}/build/mod5_mla}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
        -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math
        -fstack-protector-all -fPIC -I "${root}/include" -I "${root}/native")

"${CC}" "${cflags[@]}" -c "${root}/native/mod5_mla_provider.c" \
  -o "${scratch}/mod5_mla_provider.o"
ar rcsD "${scratch}/libeshkol_transformer_mod5_mla.a" \
  "${scratch}/mod5_mla_provider.o"

"${CC}" "${cflags[@]}" -c "${root}/native/mod5_mla_eshkol_bridge.c" \
  -o "${scratch}/mod5_mla_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_mod5_mla_bridge.a" \
  "${scratch}/mod5_mla_eshkol_bridge.o"

# Audit provider exports & undefined imports (zero allocator imports).
provider_exports="$(nm -g --defined-only "${scratch}/mod5_mla_provider.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_provider_exports="$(printf 'T %s\n' \
  et_mod5_mla_backward_f32 \
  et_mod5_mla_cache_init \
  et_mod5_mla_cache_prefill_f32 \
  et_mod5_mla_cache_reset \
  et_mod5_mla_cache_step_f32 \
  et_mod5_mla_forward_f32 \
  et_mod5_mla_full_kv_bytes \
  et_mod5_mla_kernel_provider_v1 \
  et_mod5_mla_kv_compression_ratio \
  et_mod5_mla_latent_kv_bytes)"
if [[ "${provider_exports}" != "${expected_provider_exports}" ]]; then
  printf 'unexpected MOD5 MLA provider exports:\n%s\n' "${provider_exports}" >&2
  exit 1
fi

provider_imports="$(nm -u "${scratch}/mod5_mla_provider.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|exp|\
    et_kernel_error_clear|fegetenv|fegetround|fesetenv|log|memcpy|memset|\
    snprintf|sqrt|strcmp) ;;
    *)
      printf 'unexpected MOD5 MLA provider dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${provider_imports}"

# Audit bridge exports & undefined imports (zero allocator imports).
bridge_exports="$(nm -g --defined-only "${scratch}/mod5_mla_eshkol_bridge.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_bridge_exports="$(printf 'T et_mod5_mla_aot_%s\n' \
  backward cache_init cache_prefill cache_reset cache_seq_len cache_step \
  cache_struct_bytes close dims_struct_bytes forward full_kv_bytes \
  kv_compression_ratio latent_kv_bytes load_f32 open set_dims store_f32)"
if [[ "${bridge_exports}" != "${expected_bridge_exports}" ]]; then
  printf 'unexpected MOD5 MLA bridge exports:\n%s\n' "${bridge_exports}" >&2
  exit 1
fi

bridge_imports="$(nm -u "${scratch}/mod5_mla_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|et_kernel_runtime_destroy|\
    et_kernel_runtime_discover|et_kernel_runtime_dispatch|\
    et_mod5_mla_cache_init|et_mod5_mla_cache_prefill_f32|\
    et_mod5_mla_cache_reset|et_mod5_mla_cache_step_f32|\
    et_mod5_mla_full_kv_bytes|et_mod5_mla_kernel_provider_v1|\
    et_mod5_mla_kv_compression_ratio|et_mod5_mla_latent_kv_bytes|\
    memcpy|memset|strcmp) ;;
    *)
      printf 'unexpected MOD5 MLA bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f "${scratch}/libeshkol_transformer_mod5_mla.a" "${artifact}/"
mv -f "${scratch}/libeshkol_transformer_mod5_mla_bridge.a" "${artifact}/"
mv -f "${scratch}/mod5_mla_eshkol_bridge.o" "${artifact}/"
printf 'MOD5 MLA archives and symbol audit: PASS (%s/libeshkol_transformer_mod5_mla.a)\n' "${artifact}"
