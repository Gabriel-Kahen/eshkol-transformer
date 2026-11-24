#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/radix_kv}"
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
#include "eshkol_transformer/radix_kv_abi.h"
#include "radix_kv_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_radix_kv_config>);
static_assert(std::is_standard_layout_v<et_radix_kv_stats>);
static_assert(std::is_standard_layout_v<et_radix_kv_match_result>);
static_assert(std::is_same_v<et_radix_kv_status, int32_t>);
static_assert(std::is_same_v<et_radix_kv_handle, uint64_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/radix_kv.c" \
    -o "${tdir}/radix_kv.o"
  ar rcsD "${tdir}/libeshkol_transformer_radix_kv.a" "${tdir}/radix_kv.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/radix_kv_eshkol_bridge.c" \
    -o "${tdir}/radix_kv_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_radix_kv_aot.a" \
    "${tdir}/radix_kv_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_radix_kv.a" \
    "${scratch}/second/libeshkol_transformer_radix_kv.a"
cmp "${scratch}/first/libeshkol_transformer_radix_kv_aot.a" \
    "${scratch}/second/libeshkol_transformer_radix_kv_aot.a"

# Audit defined global exports on libeshkol_transformer_radix_kv.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_radix_kv_attend_gqa_f32
T et_radix_kv_config_get
T et_radix_kv_create
T et_radix_kv_destroy
T et_radix_kv_evict_lru
T et_radix_kv_handle_info
T et_radix_kv_insert
T et_radix_kv_insert_suffix
T et_radix_kv_match_prefix
T et_radix_kv_pin
T et_radix_kv_pool_float_count
T et_radix_kv_read_kv
T et_radix_kv_reset
T et_radix_kv_root_handle
T et_radix_kv_snapshot_load
T et_radix_kv_snapshot_save
T et_radix_kv_stats_get
T et_radix_kv_unpin
T et_radix_kv_validate_config
T et_radix_kv_verify_invariants
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_radix_kv.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on radix_kv.o (no threading/dlopen/exec).
core_imports="$(nm -u "${scratch}/first/radix_kv.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|exp|fchmod|fegetround|free|fstat|fsync|malloc|\
    memcmp|memcpy|memset|mkstemp|open|read|rename|strrchr|unlink|write) ;;
    *)
      printf 'unexpected RADIX_KV core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_radix_kv_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_radix_kv_aot_attend_gqa
T et_radix_kv_aot_create
T et_radix_kv_aot_destroy
T et_radix_kv_aot_evict_lru
T et_radix_kv_aot_get_out
T et_radix_kv_aot_get_out_bits
T et_radix_kv_aot_get_read_kv
T et_radix_kv_aot_handle_depth
T et_radix_kv_aot_handle_pin_count
T et_radix_kv_aot_insert
T et_radix_kv_aot_insert_suffix
T et_radix_kv_aot_last_did_split
T et_radix_kv_aot_last_handle
T et_radix_kv_aot_last_inserted_tokens
T et_radix_kv_aot_last_kv_slot
T et_radix_kv_aot_last_matched_len
T et_radix_kv_aot_match_prefix
T et_radix_kv_aot_pin
T et_radix_kv_aot_read_kv
T et_radix_kv_aot_reset
T et_radix_kv_aot_root_handle
T et_radix_kv_aot_set_kv
T et_radix_kv_aot_set_q
T et_radix_kv_aot_set_token
T et_radix_kv_aot_snapshot_reload_env
T et_radix_kv_aot_snapshot_save_env
T et_radix_kv_aot_snapshot_tamper_check_env
T et_radix_kv_aot_stat_active_nodes
T et_radix_kv_aot_stat_evictable_tokens
T et_radix_kv_aot_stat_free_tokens
T et_radix_kv_aot_stat_lru_tick
T et_radix_kv_aot_stat_pinned_tokens
T et_radix_kv_aot_stat_total_edge_splits
T et_radix_kv_aot_stat_total_evicted_tokens
T et_radix_kv_aot_stat_total_inserted_tokens
T et_radix_kv_aot_stat_total_matched_tokens
T et_radix_kv_aot_stat_total_oom_rejections
T et_radix_kv_aot_stat_used_tokens
T et_radix_kv_aot_unpin
T et_radix_kv_aot_verify_invariants
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_radix_kv_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on radix_kv_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/radix_kv_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|et_radix_kv_*|free|fstat|getenv|malloc|memcmp|memcpy|memset|\
    open|read|strlen|unlink|write) ;;
    *)
      printf 'unexpected RADIX_KV bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_radix_kv.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_radix_kv_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_radix_kv.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_radix_kv_aot.a")"
printf 'RADIX_KV deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
