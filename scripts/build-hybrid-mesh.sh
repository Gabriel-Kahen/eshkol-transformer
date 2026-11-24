#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/hybrid_mesh}"
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

# C++17 header syntax and standard-layout check on public ABI & bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/hybrid_mesh_abi.h"
#include "hybrid_mesh_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_mesh_coords_v1>);
static_assert(std::is_standard_layout_v<et_mesh_comm_stats_v1>);
static_assert(std::is_standard_layout_v<et_mesh_model_config_v1>);
static_assert(std::is_standard_layout_v<et_mesh_optim_config_v1>);
static_assert(std::is_standard_layout_v<et_mesh_memory_report_v1>);
static_assert(std::is_standard_layout_v<et_mesh_step_info_v1>);
static_assert(std::is_same_v<et_mesh_status_t, int32_t>);
CPP

# Build both archives twice from fresh trial dirs to prove determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/hybrid_mesh.c" \
    -o "${tdir}/hybrid_mesh.o"
  ar rcsD "${tdir}/libeshkol_transformer_hybrid_mesh.a" \
    "${tdir}/hybrid_mesh.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/hybrid_mesh_eshkol_bridge.c" \
    -o "${tdir}/hybrid_mesh_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_hybrid_mesh_aot.a" \
    "${tdir}/hybrid_mesh_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_hybrid_mesh.a" \
    "${scratch}/second/libeshkol_transformer_hybrid_mesh.a"
cmp "${scratch}/first/libeshkol_transformer_hybrid_mesh_aot.a" \
    "${scratch}/second/libeshkol_transformer_hybrid_mesh_aot.a"

# Audit defined global exports on libeshkol_transformer_hybrid_mesh.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_mesh_consensus_or_u64
T et_mesh_create_fds
T et_mesh_destroy
T et_mesh_dp_allreduce_avg_f32
T et_mesh_dp_allreduce_avg_f64
T et_mesh_engine_backward
T et_mesh_engine_create
T et_mesh_engine_destroy
T et_mesh_engine_forward
T et_mesh_engine_gather_full_params
T et_mesh_engine_gather_full_params_sha256
T et_mesh_engine_get_local_grads
T et_mesh_engine_get_local_moments
T et_mesh_engine_get_local_params
T et_mesh_engine_get_memory_report
T et_mesh_engine_get_step
T et_mesh_engine_load_checkpoint
T et_mesh_engine_load_full_params
T et_mesh_engine_local_state_sha256
T et_mesh_engine_save_checkpoint
T et_mesh_engine_train_step
T et_mesh_engine_train_step_inject_nonfinite
T et_mesh_get_coords
T et_mesh_get_stats
T et_mesh_init_full_params
T et_mesh_make_microbatch
T et_mesh_open_unix
T et_mesh_param_counts
T et_mesh_reset_stats
T et_mesh_sha256_bytes
T et_mesh_tp_allgather_f32
T et_mesh_tp_allreduce_sum_f32
T et_mesh_tp_allreduce_sum_f64
T et_mesh_validate_configs
T et_mesh_validate_topology
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_hybrid_mesh.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit defined global exports on libeshkol_transformer_hybrid_mesh_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_mesh_aot_close
T et_mesh_aot_corrupt_file_byte
T et_mesh_aot_dp_allreduce_calls
T et_mesh_aot_dp_bytes_sent
T et_mesh_aot_dp_rank
T et_mesh_aot_dp_size
T et_mesh_aot_fd_grad_check
T et_mesh_aot_full_digest_byte
T et_mesh_aot_full_params
T et_mesh_aot_full_sharded_bytes
T et_mesh_aot_gather_digest
T et_mesh_aot_global_rank
T et_mesh_aot_hash_local_shard
T et_mesh_aot_init_params
T et_mesh_aot_last_clipped
T et_mesh_aot_last_fd_max_err
T et_mesh_aot_last_global_loss
T et_mesh_aot_last_grad_norm
T et_mesh_aot_last_local_loss
T et_mesh_aot_load_checkpoint
T et_mesh_aot_local_digest_byte
T et_mesh_aot_make_symlink
T et_mesh_aot_open
T et_mesh_aot_probe_tag_mismatch
T et_mesh_aot_rank_params
T et_mesh_aot_rank_sharded_bytes
T et_mesh_aot_replicated_bytes
T et_mesh_aot_save_checkpoint
T et_mesh_aot_save_shard_mark
T et_mesh_aot_shard_matches_mark
T et_mesh_aot_step
T et_mesh_aot_step_inject_nonfinite
T et_mesh_aot_tp_allgather_calls
T et_mesh_aot_tp_allreduce_calls
T et_mesh_aot_tp_bytes_sent
T et_mesh_aot_tp_rank
T et_mesh_aot_tp_sharding_ratio
T et_mesh_aot_tp_size
T et_mesh_aot_train_step
T et_mesh_aot_truncate_file
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_hybrid_mesh_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on hybrid_mesh.o (no exec, no shared memory, no dlopen).
core_imports="$(nm -u "${scratch}/first/hybrid_mesh.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    accept4|bind|calloc|clock_gettime|close|connect|exp|fcntl|free|fstat|fsync|\
    getpeername|getpid|getrandom|getsockname|getsockopt|listen|log|lstat|\
    malloc|memcmp|memcpy|memset|open|poll|pow|read|recv|rename|send|snprintf|\
    socket|sqrt|strrchr|unlink|write) ;;
    *)
      printf 'unexpected hybrid_mesh core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_hybrid_mesh.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_hybrid_mesh_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_hybrid_mesh.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_hybrid_mesh_aot.a")"
printf 'HYBRID_MESH deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
