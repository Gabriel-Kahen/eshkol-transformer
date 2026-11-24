#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/fp8_gemm}"
mkdir -p -- "${output}"
scratch="$(mktemp -d "${output}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC
  "-ffile-prefix-map=${root}=."
  -I "${root}/include" -I "${root}/native"
)

# C++17 header check on public ABI and bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/fp8_gemm_abi.h"
#include "fp8_gemm_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_fp8_delayed_scale_state_v1>);
static_assert(std::is_trivially_copyable_v<et_fp8_delayed_scale_state_v1>);
static_assert(std::is_same_v<et_fp8_status, int32_t>);
static_assert(std::is_same_v<et_fp8_format, uint32_t>);
static_assert(std::is_same_v<et_fp8_overflow_mode, uint32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/fp8_gemm.c" \
    -o "${tdir}/fp8_gemm.o"
  ar rcsD "${tdir}/libeshkol_transformer_fp8_gemm.a" "${tdir}/fp8_gemm.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/fp8_gemm_eshkol_bridge.c" \
    -o "${tdir}/fp8_gemm_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_fp8_gemm_aot.a" \
    "${tdir}/fp8_gemm_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_fp8_gemm.a" \
    "${scratch}/second/libeshkol_transformer_fp8_gemm.a"
cmp "${scratch}/first/libeshkol_transformer_fp8_gemm_aot.a" \
    "${scratch}/second/libeshkol_transformer_fp8_gemm_aot.a"

# Audit defined global exports on libeshkol_transformer_fp8_gemm.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_fp8_block_dequantize_e4m3_v1
T et_fp8_block_gemm_e4m3_backward_v1
T et_fp8_block_gemm_e4m3_v1
T et_fp8_block_quantize_e4m3_v1
T et_fp8_delayed_scale_compute
T et_fp8_delayed_scale_init
T et_fp8_delayed_scale_update
T et_fp8_delayed_scale_update_tensor
T et_fp8_e4m3_decode_scalar_v1
T et_fp8_e4m3_decode_v1
T et_fp8_e4m3_encode_scalar_v1
T et_fp8_e4m3_encode_v1
T et_fp8_e5m2_decode_scalar_v1
T et_fp8_e5m2_decode_v1
T et_fp8_e5m2_encode_scalar_v1
T et_fp8_e5m2_encode_v1
T et_fp8_per_tensor_quantize_e4m3_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_fp8_gemm.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on fp8_gemm.o (ZERO allocator, I/O, or threading imports).
core_imports="$(nm -u "${scratch}/first/fp8_gemm.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|fegetround|memcpy|memset) ;;
    *)
      printf 'unexpected FP8 core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_fp8_gemm_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_fp8_aot_block_dequantize_e4m3
T et_fp8_aot_block_gemm_e4m3
T et_fp8_aot_block_gemm_e4m3_backward
T et_fp8_aot_block_quantize_e4m3
T et_fp8_aot_delayed_scale_compute
T et_fp8_aot_delayed_scale_init
T et_fp8_aot_delayed_scale_state_bytes
T et_fp8_aot_delayed_scale_update
T et_fp8_aot_delayed_scale_update_tensor
T et_fp8_aot_e4m3_decode
T et_fp8_aot_e4m3_decode_scalar
T et_fp8_aot_e4m3_encode
T et_fp8_aot_e4m3_encode_scalar
T et_fp8_aot_e4m3_encode_scalar_bits
T et_fp8_aot_e5m2_decode
T et_fp8_aot_e5m2_decode_scalar
T et_fp8_aot_e5m2_encode
T et_fp8_aot_e5m2_encode_scalar
T et_fp8_aot_e5m2_encode_scalar_bits
T et_fp8_aot_load_f32
T et_fp8_aot_load_f32_bits
T et_fp8_aot_load_u8
T et_fp8_aot_per_tensor_quantize_e4m3
T et_fp8_aot_store_f32
T et_fp8_aot_store_f32_bits
T et_fp8_aot_store_u8
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_fp8_gemm_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on fp8_gemm_eshkol_bridge.o (ZERO allocator imports).
bridge_imports="$(nm -u "${scratch}/first/fp8_gemm_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|et_fp8_*|memcpy|memset) ;;
    *)
      printf 'unexpected FP8 bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_fp8_gemm.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_fp8_gemm_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_fp8_gemm.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_fp8_gemm_aot.a")"
printf 'FP8 GEMM deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
