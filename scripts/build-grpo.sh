#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/grpo}"
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
#include "eshkol_transformer/grpo_abi.h"
#include "grpo_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_grpo_loss_config>);
static_assert(std::is_standard_layout_v<et_grpo_loss_metrics>);
static_assert(std::is_standard_layout_v<et_grpo_verify_format_config>);
static_assert(std::is_standard_layout_v<et_grpo_verify_detail>);
static_assert(std::is_same_v<et_grpo_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/grpo.c" -o "${tdir}/grpo.o"
  ar rcsD "${tdir}/libeshkol_transformer_grpo.a" "${tdir}/grpo.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/grpo_eshkol_bridge.c" \
    -o "${tdir}/grpo_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_grpo_aot.a" \
    "${tdir}/grpo_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_grpo.a" \
    "${scratch}/second/libeshkol_transformer_grpo.a"
cmp "${scratch}/first/libeshkol_transformer_grpo_aot.a" \
    "${scratch}/second/libeshkol_transformer_grpo_aot.a"

# Audit defined global exports on libeshkol_transformer_grpo.a.
cat > "${scratch}/expected-grpo-exports" <<'EOF'
T et_grpo_group_advantages_v1
T et_grpo_group_advantages_v1_raw
T et_grpo_logprob_loss_backward_v1
T et_grpo_logprob_loss_forward_v1
T et_grpo_loss_backward_v1
T et_grpo_loss_backward_v1_raw
T et_grpo_loss_forward_v1
T et_grpo_loss_forward_v1_raw
T et_grpo_token_logprobs_backward_v1
T et_grpo_token_logprobs_backward_v1_raw
T et_grpo_token_logprobs_v1
T et_grpo_token_logprobs_v1_raw
T et_grpo_validate_loss_config
T et_grpo_verify_exact_match_v1
T et_grpo_verify_format_and_answer_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_grpo.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-grpo-exports"
diff -u "${scratch}/expected-grpo-exports" "${scratch}/actual-grpo-exports"

# Audit undefined imports on grpo.o: ZERO heap allocations, ZERO I/O/threading/exec.
grpo_imports="$(nm -u "${scratch}/first/grpo.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    bcmp|exp|expm1|fegetround|log|memcmp|memcpy|memset|sqrt) ;;
    *)
      printf 'unexpected GRPO core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${grpo_imports}"

# Audit defined global exports on libeshkol_transformer_grpo_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_grpo_aot_batch_size
T et_grpo_aot_check_negative_case
T et_grpo_aot_compute_advantages
T et_grpo_aot_create
T et_grpo_aot_destroy
T et_grpo_aot_expected_reward
T et_grpo_aot_feat_dim
T et_grpo_aot_get_advantage
T et_grpo_aot_get_bias
T et_grpo_aot_get_grad_bias
T et_grpo_aot_get_grad_weight
T et_grpo_aot_get_reward
T et_grpo_aot_get_seq_logprob
T et_grpo_aot_get_weight
T et_grpo_aot_greedy_reward
T et_grpo_aot_loss_backward
T et_grpo_aot_loss_forward
T et_grpo_aot_metric_active_tokens
T et_grpo_aot_metric_clip_fraction
T et_grpo_aot_metric_clipped_tokens
T et_grpo_aot_metric_loss
T et_grpo_aot_metric_mean_kl
T et_grpo_aot_metric_mean_reward
T et_grpo_aot_metric_pg_loss
T et_grpo_aot_policy_forward
T et_grpo_aot_score_rlvr_completion
T et_grpo_aot_set_bias
T et_grpo_aot_set_feature
T et_grpo_aot_set_loss_params
T et_grpo_aot_set_mask
T et_grpo_aot_set_reward
T et_grpo_aot_set_target
T et_grpo_aot_set_weight
T et_grpo_aot_snapshot_old
T et_grpo_aot_snapshot_ref
T et_grpo_aot_verify_exact_case
T et_grpo_aot_vocab_size
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_grpo_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on grpo_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/grpo_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|et_grpo_*|exp|free|memcpy|memset|strlen) ;;
    *)
      printf 'unexpected GRPO bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_grpo.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_grpo_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_grpo.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_grpo_aot.a")"
printf 'GRPO deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
