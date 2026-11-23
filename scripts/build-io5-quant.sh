#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${root}/build/io5_quant"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

"${CC}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math \
  -fstack-protector-all -fPIC -I "${root}/include" \
  -c "${root}/native/io5_quant_codec.c" \
  -o "${scratch}/io5_quant_codec.o"

ar rcsD "${scratch}/libeshkol_transformer_io5_quant.a" \
  "${scratch}/io5_quant_codec.o"

"${CC}" -shared -Wl,--build-id=none \
  -Wl,-soname,libeshkol_transformer_io5_quant.so \
  "${scratch}/io5_quant_codec.o" \
  -o "${scratch}/libeshkol_transformer_io5_quant.so"

exports="$(nm -g --defined-only "${scratch}/libeshkol_transformer_io5_quant.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected="$(cat <<'EOF'
T et_io5_bf16_bits_to_f32_bits
T et_io5_decode_bf16
T et_io5_decode_f16
T et_io5_dequantize_q8_0
T et_io5_encode_bf16
T et_io5_encode_f16
T et_io5_f16_bits_to_f32_bits
T et_io5_f32_bits_to_bf16_bits
T et_io5_f32_bits_to_f16_bits
T et_io5_q8_0_byte_size
T et_io5_quant_abi_version
T et_io5_quantize_q8_0
T et_io5_validate_f32_finite
EOF
)"
if [[ "${exports}" != "${expected}" ]]; then
  printf 'unexpected IO5 quant exports:\n%s\n' "${exports}" >&2
  exit 1
fi

# Reject allocator, foreign-runtime, math-lib, and threading imports.
imports="$(nm -u "${scratch}/io5_quant_codec.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|memcpy|memset) ;;
    *)
      printf 'unexpected IO5 quant dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

mv -f "${scratch}/libeshkol_transformer_io5_quant.a" "${artifact}/"
mv -f "${scratch}/libeshkol_transformer_io5_quant.so" "${artifact}/"

a_sha="$(sha256sum "${artifact}/libeshkol_transformer_io5_quant.a" | awk '{print $1}')"
so_sha="$(sha256sum "${artifact}/libeshkol_transformer_io5_quant.so" | awk '{print $1}')"
printf 'IO5 quant deterministic archive SHA-256: %s\n' "${a_sha}"
printf 'IO5 quant deterministic shared lib SHA-256: %s\n' "${so_sha}"
printf 'IO5 quant build and symbol audit: PASS (%s)\n' "${artifact}"
