#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/mtp_head}"
mkdir -p -- "${output}"
scratch="$(mktemp -d "${output}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
cflags=(
  -std=c11 "${warnings[@]}" -O2 -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math
  -fstack-protector-all -fPIC
  "-ffile-prefix-map=${root}=."
  -I "${root}/include" -I "${root}/native"
)

# C++17 header syntax and standard-layout verification.
"${CXX}" -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/mtp_head_abi.h"
#include "mtp_head_eshkol_bridge.h"
#include <type_traits>
static_assert(ET_MTP_ABI_MAJOR == 1u);
static_assert(ET_MTP_OK == 0);
static_assert(std::is_same_v<et_mtp_status_t, int32_t>);
static_assert(std::is_standard_layout_v<et_mtp_config_v1>);
static_assert(std::is_standard_layout_v<et_mtp_layout_v1>);
static_assert(std::is_standard_layout_v<et_mtp_loss_stats_v1>);
static_assert(std::is_standard_layout_v<et_mtp_step_stats_v1>);
static_assert(std::is_standard_layout_v<et_mtp_rng_state_v1>);
static_assert(std::is_standard_layout_v<et_mtp_verify_result_v1>);
CPP

# Build both archives twice from fresh directories to prove byte-identical determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/mtp_head.c" \
    -o "${tdir}/mtp_head.o"
  ar rcsD "${tdir}/libeshkol_transformer_mtp_head.a" \
    "${tdir}/mtp_head.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/mtp_head_eshkol_bridge.c" \
    -o "${tdir}/mtp_head_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_mtp_head_aot.a" \
    "${tdir}/mtp_head_eshkol_bridge.o"
done

cmp -- "${scratch}/first/libeshkol_transformer_mtp_head.a" \
       "${scratch}/second/libeshkol_transformer_mtp_head.a"
cmp -- "${scratch}/first/libeshkol_transformer_mtp_head_aot.a" \
       "${scratch}/second/libeshkol_transformer_mtp_head_aot.a"

# Audit defined global exports on libeshkol_transformer_mtp_head.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_mtp_adamw_step_v1
T et_mtp_checkpoint_load_v1
T et_mtp_checkpoint_save_v1
T et_mtp_compute_layout_v1
T et_mtp_draft_verify_v1
T et_mtp_forward_backward_v1
T et_mtp_forward_v1
T et_mtp_main_logits_v1
T et_mtp_model_create_v1
T et_mtp_model_destroy_v1
T et_mtp_model_get_config_v1
T et_mtp_model_get_layout_v1
T et_mtp_model_get_moments_v1
T et_mtp_model_get_params_v1
T et_mtp_model_get_step_v1
T et_mtp_model_init_weights_v1
T et_mtp_model_set_params_v1
T et_mtp_model_state_sha256_v1
T et_mtp_sha256_bytes_v1
T et_mtp_validate_config_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_mtp_head.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit defined global exports on libeshkol_transformer_mtp_head_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_mtp_aot_close
T et_mtp_aot_compute_state_digest
T et_mtp_aot_copy_file
T et_mtp_aot_corrupt_file_byte
T et_mtp_aot_digest_byte
T et_mtp_aot_fd_grad_check
T et_mtp_aot_init_sequence
T et_mtp_aot_init_weights
T et_mtp_aot_last_fd_checked
T et_mtp_aot_last_fd_h0_max_err
T et_mtp_aot_last_fd_param_max_err
T et_mtp_aot_last_grad_norm
T et_mtp_aot_last_loss_depth
T et_mtp_aot_last_loss_main
T et_mtp_aot_last_loss_mtp_avg
T et_mtp_aot_last_loss_total
T et_mtp_aot_load_checkpoint
T et_mtp_aot_make_symlink
T et_mtp_aot_open
T et_mtp_aot_probe_ckpt_config_mismatch
T et_mtp_aot_probe_fp_env_trap
T et_mtp_aot_probe_invalid_depth
T et_mtp_aot_probe_nonfinite_h0
T et_mtp_aot_probe_oob_token
T et_mtp_aot_probe_short_seq
T et_mtp_aot_save_checkpoint
T et_mtp_aot_spec_exact_match
T et_mtp_aot_spec_greedy_verify
T et_mtp_aot_spec_tokens_per_step
T et_mtp_aot_spec_total_emitted
T et_mtp_aot_spec_verify_steps
T et_mtp_aot_step
T et_mtp_aot_total_params
T et_mtp_aot_train_step
T et_mtp_aot_truncate_file
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_mtp_head_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

mv -f -- "${scratch}/first/libeshkol_transformer_mtp_head.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_mtp_head_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_mtp_head.a" | awk '{print $1}')"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_mtp_head_aot.a" | awk '{print $1}')"
printf 'MTP_HEAD deterministic archives and symbol audit: PASS (core=%s aot=%s)\n' \
  "${core_sha}" "${bridge_sha}"
