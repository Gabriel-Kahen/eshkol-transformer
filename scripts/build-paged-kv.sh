#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/paged_kv}"
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
#include "eshkol_transformer/paged_kv_abi.h"
#include "paged_kv_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_paged_kv_config>);
static_assert(std::is_standard_layout_v<et_paged_kv_stats>);
static_assert(std::is_same_v<et_paged_kv_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/paged_kv.c" \
    -o "${tdir}/paged_kv.o"
  ar rcsD "${tdir}/libeshkol_transformer_paged_kv.a" "${tdir}/paged_kv.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/paged_kv_eshkol_bridge.c" \
    -o "${tdir}/paged_kv_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_paged_kv_aot.a" \
    "${tdir}/paged_kv_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_paged_kv.a" \
    "${scratch}/second/libeshkol_transformer_paged_kv.a"
cmp "${scratch}/first/libeshkol_transformer_paged_kv_aot.a" \
    "${scratch}/second/libeshkol_transformer_paged_kv_aot.a"

# Audit defined global exports on libeshkol_transformer_paged_kv.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_paged_kv_append_token
T et_paged_kv_append_token_raw
T et_paged_kv_append_tokens
T et_paged_kv_append_tokens_raw
T et_paged_kv_block_refcount
T et_paged_kv_config_get
T et_paged_kv_create
T et_paged_kv_create_raw
T et_paged_kv_create_with_prefix_context_v1
T et_paged_kv_decode_attention_v1
T et_paged_kv_decode_attention_v1_raw
T et_paged_kv_destroy
T et_paged_kv_hash_token_block
T et_paged_kv_hash_token_block_with_context_v1
T et_paged_kv_pool_float_count
T et_paged_kv_prefix_cache_attach
T et_paged_kv_prefix_cache_attach_with_context_v1
T et_paged_kv_prefix_cache_register
T et_paged_kv_prefix_cache_register_with_context_v1
T et_paged_kv_read_token
T et_paged_kv_read_token_raw
T et_paged_kv_reset
T et_paged_kv_seq_alloc
T et_paged_kv_seq_block_table
T et_paged_kv_seq_fork
T et_paged_kv_seq_free
T et_paged_kv_seq_len
T et_paged_kv_seq_truncate
T et_paged_kv_stats_get
T et_paged_kv_validate_config
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_paged_kv.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on paged_kv.o (no threading/dlopen/exec/IO).
core_imports="$(nm -u "${scratch}/first/paged_kv.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|exp|fegetround|free|memcmp|memcpy|memset) ;;
    *)
      printf 'unexpected PAGED_KV core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_paged_kv_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_paged_kv_aot_append_token
T et_paged_kv_aot_block_refcount
T et_paged_kv_aot_create
T et_paged_kv_aot_decode_attention
T et_paged_kv_aot_destroy
T et_paged_kv_aot_get_out
T et_paged_kv_aot_get_out_bits
T et_paged_kv_aot_hash_block
T et_paged_kv_aot_prefix_attach
T et_paged_kv_aot_prefix_register
T et_paged_kv_aot_read_kv
T et_paged_kv_aot_reset
T et_paged_kv_aot_seq_alloc
T et_paged_kv_aot_seq_fork
T et_paged_kv_aot_seq_free
T et_paged_kv_aot_seq_len
T et_paged_kv_aot_seq_num_blocks
T et_paged_kv_aot_seq_phys_block
T et_paged_kv_aot_seq_truncate
T et_paged_kv_aot_set_k
T et_paged_kv_aot_set_q
T et_paged_kv_aot_set_token_id
T et_paged_kv_aot_set_v
T et_paged_kv_aot_stat_active_seqs
T et_paged_kv_aot_stat_cow_copies
T et_paged_kv_aot_stat_free_blocks
T et_paged_kv_aot_stat_prefix_cached
T et_paged_kv_aot_stat_prefix_hits
T et_paged_kv_aot_stat_shared_blocks
T et_paged_kv_aot_stat_total_blocks
T et_paged_kv_aot_stat_used_blocks
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_paged_kv_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on paged_kv_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/paged_kv_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|et_paged_kv_*|free|memcpy|memset) ;;
    *)
      printf 'unexpected PAGED_KV bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_paged_kv.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_paged_kv_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_paged_kv.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_paged_kv_aot.a")"
printf 'PAGED_KV deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
