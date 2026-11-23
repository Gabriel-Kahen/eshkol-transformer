#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
artifact="${root}/build/mp4_aot"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math \
  -fstack-protector-all -fPIC -I "${root}/include" -I "${root}/native" \
  -c "${root}/native/mp4_eshkol_bridge.c" \
  -o "${scratch}/mp4_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_mp4_aot.a" \
  "${scratch}/mp4_eshkol_bridge.o"

exports="$(nm -g --defined-only "${scratch}/libeshkol_transformer_mp4_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected="$(cat <<'EOF'
T et_mp4_aot_cast_bf16_to_f32
T et_mp4_aot_cast_f16_to_f32
T et_mp4_aot_cast_f32_to_bf16
T et_mp4_aot_cast_f32_to_f16
T et_mp4_aot_close
T et_mp4_aot_element_bytes
T et_mp4_aot_grad_unscale_bf16
T et_mp4_aot_grad_unscale_f16
T et_mp4_aot_load_f32
T et_mp4_aot_load_f32_bits
T et_mp4_aot_load_u16
T et_mp4_aot_load_u8
T et_mp4_aot_open
T et_mp4_aot_scaler_bytes
T et_mp4_aot_scaler_get_growth_tracker
T et_mp4_aot_scaler_get_overflow_count
T et_mp4_aot_scaler_get_scale
T et_mp4_aot_scaler_init
T et_mp4_aot_scaler_scale
T et_mp4_aot_scaler_scale_f32
T et_mp4_aot_scaler_step
T et_mp4_aot_scaler_unscale_bf16
T et_mp4_aot_scaler_unscale_f16
T et_mp4_aot_store_f32
T et_mp4_aot_store_f32_bits
T et_mp4_aot_store_u16
T et_mp4_aot_store_u8
T et_mp4_aot_tensor_bytes
EOF
)"
if [[ "${exports}" != "${expected}" ]]; then
  printf 'unexpected MP4 AOT bridge exports:\n%s\n' "${exports}" >&2
  exit 1
fi

# Reject allocator, foreign-runtime, and threading imports in the bridge object.
imports="$(nm -u "${scratch}/mp4_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    et_kernel_runtime_destroy|et_kernel_runtime_discover|et_kernel_runtime_dispatch|\
    et_mp4_kernel_provider_v1|et_mp4_loss_scaler_init|et_mp4_loss_scaler_update|\
    memcpy|strcmp) ;;
    *)
      printf 'unexpected MP4 AOT bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

mv -f "${scratch}/libeshkol_transformer_mp4_aot.a" "${artifact}/"
printf 'MP4 AOT deterministic archive and symbol audit: PASS (%s)\n' \
  "${artifact}/libeshkol_transformer_mp4_aot.a"
