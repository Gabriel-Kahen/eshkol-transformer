#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/fsdp_zero}"
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
#include "eshkol_transformer/fsdp_zero_abi.h"
#include "fsdp_zero_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_fsdp_config_v1>);
static_assert(std::is_standard_layout_v<et_fsdp_memory_report_v1>);
static_assert(std::is_standard_layout_v<et_fsdp_comm_stats_v1>);
static_assert(std::is_standard_layout_v<et_fsdp_step_info_v1>);
static_assert(std::is_same_v<et_fsdp_status_t, int32_t>);
static_assert(std::is_same_v<et_zero_stage_t, uint32_t>);
CPP

# Build both archives twice from fresh trial dirs to prove determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/fsdp_zero.c" \
    -o "${tdir}/fsdp_zero.o"
  ar rcsD "${tdir}/libeshkol_transformer_fsdp_zero.a" \
    "${tdir}/fsdp_zero.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/fsdp_zero_eshkol_bridge.c" \
    -o "${tdir}/fsdp_zero_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_fsdp_zero_aot.a" \
    "${tdir}/fsdp_zero_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_fsdp_zero.a" \
    "${scratch}/second/libeshkol_transformer_fsdp_zero.a"
cmp "${scratch}/first/libeshkol_transformer_fsdp_zero_aot.a" \
    "${scratch}/second/libeshkol_transformer_fsdp_zero_aot.a"

# Audit defined global exports on libeshkol_transformer_fsdp_zero.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_fsdp_allgather_f32
T et_fsdp_allreduce_avg_f32
T et_fsdp_allreduce_sum_f64
T et_fsdp_comm_create_fds
T et_fsdp_comm_destroy
T et_fsdp_comm_get_stats
T et_fsdp_comm_open_unix
T et_fsdp_comm_rank
T et_fsdp_comm_world_size
T et_fsdp_compute_padded_len
T et_fsdp_engine_create
T et_fsdp_engine_destroy
T et_fsdp_engine_gather_full_params_sha256
T et_fsdp_engine_gather_params
T et_fsdp_engine_get_grad_shard
T et_fsdp_engine_get_memory_report
T et_fsdp_engine_get_moments_shard
T et_fsdp_engine_get_param_shard
T et_fsdp_engine_get_step
T et_fsdp_engine_load_checkpoint
T et_fsdp_engine_load_initial_params
T et_fsdp_engine_poison_ephemeral
T et_fsdp_engine_save_checkpoint
T et_fsdp_engine_step
T et_fsdp_mlp_forward_backward
T et_fsdp_mlp_init_params
T et_fsdp_mlp_make_batch
T et_fsdp_mlp_num_params
T et_fsdp_reduce_scatter_avg_f32
T et_fsdp_sha256_bytes
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_fsdp_zero.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit defined global exports on libeshkol_transformer_fsdp_zero_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_fsdp_aot_close
T et_fsdp_aot_comm_bytes_recv
T et_fsdp_aot_comm_bytes_sent
T et_fsdp_aot_corrupt_file_byte
T et_fsdp_aot_ephemeral_poison_verified
T et_fsdp_aot_fd_grad_check
T et_fsdp_aot_full_digest_byte
T et_fsdp_aot_gather_digest
T et_fsdp_aot_hash_local_shard
T et_fsdp_aot_init_params
T et_fsdp_aot_last_clip_scale
T et_fsdp_aot_last_clipped
T et_fsdp_aot_last_fd_max_err
T et_fsdp_aot_last_grad_norm
T et_fsdp_aot_last_loss
T et_fsdp_aot_load_checkpoint
T et_fsdp_aot_make_symlink
T et_fsdp_aot_num_params
T et_fsdp_aot_open
T et_fsdp_aot_opt_reduction_ratio
T et_fsdp_aot_padded_params
T et_fsdp_aot_persistent_grad_bytes
T et_fsdp_aot_persistent_opt_bytes
T et_fsdp_aot_persistent_param_bytes
T et_fsdp_aot_persistent_total_bytes
T et_fsdp_aot_probe_tag_mismatch
T et_fsdp_aot_rank
T et_fsdp_aot_save_checkpoint
T et_fsdp_aot_save_shard_digest_mark
T et_fsdp_aot_shard_digest_matches_mark
T et_fsdp_aot_shard_len
T et_fsdp_aot_stage
T et_fsdp_aot_step
T et_fsdp_aot_step_inject_nonfinite
T et_fsdp_aot_total_reduction_ratio
T et_fsdp_aot_train_step
T et_fsdp_aot_truncate_file
T et_fsdp_aot_unsharded_baseline_bytes
T et_fsdp_aot_world_size
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_fsdp_zero_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

mv -f -- "${scratch}/first/libeshkol_transformer_fsdp_zero.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_fsdp_zero_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_fsdp_zero.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_fsdp_zero_aot.a")"
printf 'FSDP_ZERO deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
