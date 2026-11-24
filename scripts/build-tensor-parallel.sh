#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/tensor_parallel}"
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
#include "eshkol_transformer/tensor_parallel_abi.h"
#include "tensor_parallel_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_tp_collective_stats_t>);
static_assert(std::is_standard_layout_v<et_tp_swiglu_config_t>);
static_assert(std::is_standard_layout_v<et_tp_gqa_config_t>);
static_assert(std::is_standard_layout_v<et_tp_block_config_t>);
static_assert(std::is_standard_layout_v<et_tp_block_memory_stats_t>);
static_assert(std::is_standard_layout_v<et_tp_block_weights_t>);
static_assert(std::is_standard_layout_v<et_tp_block_grads_t>);
static_assert(std::is_standard_layout_v<et_tp_block_cache_t>);
static_assert(std::is_same_v<et_tp_status_t, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/tensor_parallel.c" \
    -o "${tdir}/tensor_parallel.o"
  ar rcsD "${tdir}/libeshkol_transformer_tensor_parallel.a" \
    "${tdir}/tensor_parallel.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/tensor_parallel_eshkol_bridge.c" \
    -o "${tdir}/tensor_parallel_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_tensor_parallel_aot.a" \
    "${tdir}/tensor_parallel_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_tensor_parallel.a" \
    "${scratch}/second/libeshkol_transformer_tensor_parallel.a"
cmp "${scratch}/first/libeshkol_transformer_tensor_parallel_aot.a" \
    "${scratch}/second/libeshkol_transformer_tensor_parallel_aot.a"

# Audit defined global exports on libeshkol_transformer_tensor_parallel.a.
cat > "${scratch}/expected-tp-exports" <<'EOF'
T et_tp_allgather_f32
T et_tp_allreduce_sum_f32
T et_tp_block_backward_v1
T et_tp_block_forward_v1
T et_tp_block_gather_weights_v1
T et_tp_block_memory_stats_v1
T et_tp_block_shard_weights_v1
T et_tp_block_validate_config
T et_tp_column_linear_backward_v1
T et_tp_column_linear_forward_v1
T et_tp_gqa_backward_v1
T et_tp_gqa_forward_v1
T et_tp_gqa_validate_config
T et_tp_group_create
T et_tp_group_destroy
T et_tp_group_get_stats
T et_tp_group_rank
T et_tp_group_reset_stats
T et_tp_group_set_timeout_ms
T et_tp_group_world_size
T et_tp_rmsnorm_backward_v1
T et_tp_rmsnorm_forward_v1
T et_tp_row_linear_backward_v1
T et_tp_row_linear_forward_v1
T et_tp_swiglu_backward_v1
T et_tp_swiglu_forward_v1
T et_tp_swiglu_validate_config
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_tensor_parallel.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-tp-exports"
diff -u "${scratch}/expected-tp-exports" "${scratch}/actual-tp-exports"

# Audit defined global exports on libeshkol_transformer_tensor_parallel_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_tp_aot_allgather_calls
T et_tp_aot_allreduce_calls
T et_tp_aot_backward
T et_tp_aot_check_negative_case
T et_tp_aot_forward
T et_tp_aot_full_sharded_weight_bytes
T et_tp_aot_gather_grads
T et_tp_aot_gather_weights
T et_tp_aot_get_dx
T et_tp_aot_get_gathered_grad
T et_tp_aot_get_gathered_weight
T et_tp_aot_get_input
T et_tp_aot_get_output
T et_tp_aot_get_weight
T et_tp_aot_ipc_allgather_calls
T et_tp_aot_ipc_allreduce_calls
T et_tp_aot_max_dx_diff
T et_tp_aot_max_grad_diff
T et_tp_aot_max_output_diff
T et_tp_aot_max_weight_diff
T et_tp_aot_mse_loss
T et_tp_aot_output_count
T et_tp_aot_param_count
T et_tp_aot_rank_sharded_weight_bytes
T et_tp_aot_replicated_weight_bytes
T et_tp_aot_reset_stats
T et_tp_aot_session_create
T et_tp_aot_session_destroy
T et_tp_aot_set_input
T et_tp_aot_set_weight
T et_tp_aot_sgd_step
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_tensor_parallel_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on tensor_parallel.o (no shared memory, no exec).
tp_imports="$(nm -u "${scratch}/first/tensor_parallel.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|clock_gettime|close|exp|fcntl|free|fstat|getpeername|getsockname|\
    getsockopt|malloc|memcpy|memset|poll|recv|send|shutdown|sqrt) ;;
    *)
      printf 'unexpected TP core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${tp_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_tensor_parallel.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_tensor_parallel_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_tensor_parallel.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_tensor_parallel_aot.a")"
printf 'TP deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
