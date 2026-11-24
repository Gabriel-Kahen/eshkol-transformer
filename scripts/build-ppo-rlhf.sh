#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/ppo_rlhf}"
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

# C++17 standard-layout ABI header check.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/ppo_rlhf_abi.h"
#include "ppo_rlhf_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_ppo_loss_config>);
static_assert(std::is_standard_layout_v<et_ppo_loss_metrics>);
static_assert(std::is_standard_layout_v<et_ppo_adamw_config>);
static_assert(std::is_standard_layout_v<et_ppo_trainer_config>);
static_assert(std::is_standard_layout_v<et_ppo_rollout_report>);
static_assert(std::is_standard_layout_v<et_ppo_checkpoint_meta>);
static_assert(std::is_same_v<et_ppo_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/ppo_rlhf.c" -o "${tdir}/ppo_rlhf.o"
  ar rcsD "${tdir}/libeshkol_transformer_ppo_rlhf.a" "${tdir}/ppo_rlhf.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/ppo_rlhf_eshkol_bridge.c" \
    -o "${tdir}/ppo_rlhf_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_ppo_rlhf_aot.a" \
    "${tdir}/ppo_rlhf_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_ppo_rlhf.a" \
    "${scratch}/second/libeshkol_transformer_ppo_rlhf.a"
cmp "${scratch}/first/libeshkol_transformer_ppo_rlhf_aot.a" \
    "${scratch}/second/libeshkol_transformer_ppo_rlhf_aot.a"

# Audit defined global exports on libeshkol_transformer_ppo_rlhf.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_ppo_actor_critic_forward_v1
T et_ppo_adamw_step_v1
T et_ppo_checkpoint_load_v1
T et_ppo_checkpoint_save_v1
T et_ppo_gae_v1
T et_ppo_kl_shaped_rewards_v1
T et_ppo_loss_backward_v1
T et_ppo_loss_forward_v1
T et_ppo_model_param_count
T et_ppo_ref_param_count
T et_ppo_ref_policy_logprobs_v1
T et_ppo_token_logprobs_v1
T et_ppo_train_rollout_v1
T et_ppo_validate_loss_config
T et_ppo_validate_trainer_config
T et_ppo_weight_sha256_v1
T et_ppo_whiten_advantages_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_ppo_rlhf.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on ppo_rlhf.o: ZERO malloc/calloc/realloc/free!
core_imports="$(nm -u "${scratch}/first/ppo_rlhf.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    bcmp|close|exp|fchmod|fegetround|fstat|fsync|getpid|log|\
    memcmp|memcpy|memset|open|read|rename|snprintf|sqrt|strlen|strrchr|\
    unlink|write) ;;
    *)
      printf 'unexpected PPO core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_ppo_rlhf_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_ppo_aot_check_ckpt_negative
T et_ppo_aot_check_negative_case
T et_ppo_aot_configure
T et_ppo_aot_create
T et_ppo_aot_destroy
T et_ppo_aot_env_phase
T et_ppo_aot_expected_env_reward
T et_ppo_aot_fd_bt
T et_ppo_aot_fd_get_d_logit
T et_ppo_aot_fd_get_d_value
T et_ppo_aot_fd_get_logit
T et_ppo_aot_fd_get_value
T et_ppo_aot_fd_loss_backward
T et_ppo_aot_fd_loss_forward
T et_ppo_aot_fd_set_logit
T et_ppo_aot_fd_set_value
T et_ppo_aot_fd_total_loss
T et_ppo_aot_fd_vocab
T et_ppo_aot_init_task
T et_ppo_aot_load_env_ckpt
T et_ppo_aot_param_sha256_word
T et_ppo_aot_ref_sha256_word
T et_ppo_aot_refresh_digests
T et_ppo_aot_report_approx_kl
T et_ppo_aot_report_clip_fraction
T et_ppo_aot_report_early_stopped
T et_ppo_aot_report_entropy
T et_ppo_aot_report_epochs_completed
T et_ppo_aot_report_explained_variance
T et_ppo_aot_report_mean_env_reward
T et_ppo_aot_report_mean_shaped_reward
T et_ppo_aot_report_policy_loss
T et_ppo_aot_report_total_loss
T et_ppo_aot_report_value_loss
T et_ppo_aot_sample_rollout
T et_ppo_aot_save_env_ckpt
T et_ppo_aot_set_target_kl
T et_ppo_aot_setup_fd_case
T et_ppo_aot_step
T et_ppo_aot_train_rollout
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_ppo_rlhf_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on ppo_rlhf_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/ppo_rlhf_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|et_ppo_*|exp|fegetround|fesetround|free|getenv|\
    memcpy|memset|open|read|strcmp|symlink|unlink|write) ;;
    *)
      printf 'unexpected PPO bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_ppo_rlhf.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_ppo_rlhf_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_ppo_rlhf.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_ppo_rlhf_aot.a")"
printf 'PPO-RLHF deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
