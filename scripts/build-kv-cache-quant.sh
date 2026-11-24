#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/kv_cache_quant}"
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
#include "eshkol_transformer/kv_cache_quant_abi.h"
#include "kv_cache_quant_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_kvq_config>);
static_assert(std::is_trivially_copyable_v<et_kvq_config>);
static_assert(std::is_standard_layout_v<et_kvq_memory_stats>);
static_assert(std::is_trivially_copyable_v<et_kvq_memory_stats>);
static_assert(std::is_same_v<et_kvq_format, uint32_t>);
static_assert(std::is_same_v<et_kvq_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/kv_cache_quant.c" \
    -o "${tdir}/kv_cache_quant.o"
  ar rcsD "${tdir}/libeshkol_transformer_kv_cache_quant.a" \
    "${tdir}/kv_cache_quant.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/kv_cache_quant_eshkol_bridge.c" \
    -o "${tdir}/kv_cache_quant_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_kv_cache_quant_aot.a" \
    "${tdir}/kv_cache_quant_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_kv_cache_quant.a" \
    "${scratch}/second/libeshkol_transformer_kv_cache_quant.a"
cmp "${scratch}/first/libeshkol_transformer_kv_cache_quant_aot.a" \
    "${scratch}/second/libeshkol_transformer_kv_cache_quant_aot.a"

# Audit defined global exports on libeshkol_transformer_kv_cache_quant.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_kvq_append_token_v1
T et_kvq_cache_config_get
T et_kvq_cache_create
T et_kvq_cache_destroy
T et_kvq_cache_layer_seq_len
T et_kvq_cache_load_snapshot
T et_kvq_cache_memory_stats
T et_kvq_cache_reset
T et_kvq_cache_save_snapshot
T et_kvq_cache_truncate
T et_kvq_decode_gqa_v1
T et_kvq_dequant_slice_v1
T et_kvq_fp8_e4m3_decode_scalar
T et_kvq_fp8_e4m3_dequantize_vec
T et_kvq_fp8_e4m3_encode_scalar
T et_kvq_fp8_e4m3_quantize_vec
T et_kvq_int8_sym_dequantize_vec
T et_kvq_int8_sym_quantize_vec
T et_kvq_read_raw_vector
T et_kvq_storage_bytes_required
T et_kvq_validate_config
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_kv_cache_quant.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on kv_cache_quant.o (no threading/dlopen/exec/network).
core_imports="$(nm -u "${scratch}/first/kv_cache_quant.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|exp|fchmod|fegetround|free|fstat|fstatat|fsync|\
    getrandom|malloc|memcmp|memcpy|memset|open|openat|read|renameat|\
    strlen|strrchr|unlinkat|write) ;;
    *)
      printf 'unexpected KVQ core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_kv_cache_quant_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_kvq_aot_append_token
T et_kvq_aot_check_fp_env_trap
T et_kvq_aot_copy_dequant_to_fp32
T et_kvq_aot_create
T et_kvq_aot_decode_gqa
T et_kvq_aot_dequant_slice
T et_kvq_aot_destroy
T et_kvq_aot_get_deq_k
T et_kvq_aot_get_deq_v
T et_kvq_aot_get_out
T et_kvq_aot_get_out_bits
T et_kvq_aot_layer_seq_len
T et_kvq_aot_load_snapshot_slot
T et_kvq_aot_reset
T et_kvq_aot_save_snapshot_slot
T et_kvq_aot_set_k
T et_kvq_aot_set_k_bits
T et_kvq_aot_set_q
T et_kvq_aot_set_q_bits
T et_kvq_aot_set_v
T et_kvq_aot_stat_compression_ratio
T et_kvq_aot_stat_fp32_bytes
T et_kvq_aot_stat_quant_bytes
T et_kvq_aot_tamper_snapshot_slot
T et_kvq_aot_truncate
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_kv_cache_quant_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on kv_cache_quant_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/kv_cache_quant_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|et_kvq_*|fegetround|fesetround|free|fstat|getenv|\
    memcpy|memset|open|pread|pwrite|snprintf) ;;
    *)
      printf 'unexpected KVQ bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_kv_cache_quant.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_kv_cache_quant_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_kv_cache_quant.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_kv_cache_quant_aot.a")"
printf 'KVQ deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
