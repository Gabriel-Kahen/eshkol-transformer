#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/w4a32_quant}"
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

# C++17 header syntax and layout contract check.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/w4a32_quant_abi.h"
#include "w4a32_quant_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_w4a32_container_meta>);
static_assert(std::is_same_v<et_w4a32_status, int32_t>);
static_assert(std::is_same_v<et_w4a32_format, uint32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/w4a32_quant.c" \
    -o "${tdir}/w4a32_quant.o"
  ar rcsD "${tdir}/libeshkol_transformer_w4a32_quant.a" \
    "${tdir}/w4a32_quant.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/w4a32_quant_eshkol_bridge.c" \
    -o "${tdir}/w4a32_quant_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_w4a32_quant_aot.a" \
    "${tdir}/w4a32_quant_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_w4a32_quant.a" \
    "${scratch}/second/libeshkol_transformer_w4a32_quant.a"
cmp "${scratch}/first/libeshkol_transformer_w4a32_quant_aot.a" \
    "${scratch}/second/libeshkol_transformer_w4a32_quant_aot.a"

# Audit defined global exports on libeshkol_transformer_w4a32_quant.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_w4a32_asym_dequantize_v1
T et_w4a32_asym_gemm_v1
T et_w4a32_asym_quantize_v1
T et_w4a32_container_load
T et_w4a32_container_parse
T et_w4a32_container_save
T et_w4a32_container_serialize
T et_w4a32_container_size
T et_w4a32_f16_to_f32
T et_w4a32_f32_gemm_v1
T et_w4a32_f32_to_f16
T et_w4a32_packed_bytes
T et_w4a32_q4_0_dequantize_v1
T et_w4a32_q4_0_gemm_v1
T et_w4a32_q4_0_quantize_v1
T et_w4a32_sha256_bytes
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_w4a32_quant.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on w4a32_quant.o (no threading/dlopen/exec/stdio).
core_imports="$(nm -u "${scratch}/first/w4a32_quant.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    close|fegetround|floor|free|fstat|fsync|malloc|memcmp|memcpy|memset|\
    mkstemp|open|read|rename|strrchr|unlink|write) ;;
    *)
      printf 'unexpected W4A32 core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_w4a32_quant_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_w4a32_aot_b1_count
T et_w4a32_aot_b2_count
T et_w4a32_aot_check_negative
T et_w4a32_aot_compute_packed_sha256
T et_w4a32_aot_compute_y_sha256
T et_w4a32_aot_create
T et_w4a32_aot_dequantize_all
T et_w4a32_aot_destroy
T et_w4a32_aot_env_phase
T et_w4a32_aot_f32_weight_bytes
T et_w4a32_aot_get_w1
T et_w4a32_aot_get_w1_deq
T et_w4a32_aot_get_w2
T et_w4a32_aot_get_w2_deq
T et_w4a32_aot_get_y_fused
T et_w4a32_aot_get_y_ref
T et_w4a32_aot_load_env_pair
T et_w4a32_aot_mlp_forward_deq_ref
T et_w4a32_aot_mlp_forward_fused
T et_w4a32_aot_packed_sha256_word
T et_w4a32_aot_packed_weight_bytes
T et_w4a32_aot_quantize_all
T et_w4a32_aot_save_env_pair
T et_w4a32_aot_set_b1
T et_w4a32_aot_set_b2
T et_w4a32_aot_set_w1
T et_w4a32_aot_set_w2
T et_w4a32_aot_set_x
T et_w4a32_aot_w1_count
T et_w4a32_aot_w2_count
T et_w4a32_aot_x_count
T et_w4a32_aot_y_count
T et_w4a32_aot_y_fused_bits
T et_w4a32_aot_y_sha256_word
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_w4a32_quant_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on w4a32_quant_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/w4a32_quant_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|et_w4a32_*|free|getenv|malloc|memcpy|memset|open|read|\
    snprintf|strcmp|symlink|unlink|write) ;;
    *)
      printf 'unexpected W4A32 bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_w4a32_quant.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_w4a32_quant_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_w4a32_quant.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_w4a32_quant_aot.a")"
printf 'W4A32 deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
