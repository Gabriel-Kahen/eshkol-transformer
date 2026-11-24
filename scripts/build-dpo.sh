#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/dpo}"
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
#include "eshkol_transformer/dpo_alignment_abi.h"
#include "dpo_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_dpo_config>);
static_assert(std::is_standard_layout_v<et_dpo_telemetry>);
static_assert(std::is_same_v<et_dpo_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/dpo_alignment.c" \
    -o "${tdir}/dpo_alignment.o"
  ar rcsD "${tdir}/libeshkol_transformer_dpo.a" "${tdir}/dpo_alignment.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/dpo_eshkol_bridge.c" \
    -o "${tdir}/dpo_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_dpo_aot.a" \
    "${tdir}/dpo_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_dpo.a" \
    "${scratch}/second/libeshkol_transformer_dpo.a"
cmp "${scratch}/first/libeshkol_transformer_dpo_aot.a" \
    "${scratch}/second/libeshkol_transformer_dpo_aot.a"

# Audit defined global exports on libeshkol_transformer_dpo.a.
cat > "${scratch}/expected-dpo-exports" <<'EOF'
T et_dpo_forward
T et_dpo_forward_backward
T et_dpo_forward_backward_raw
T et_dpo_forward_raw
T et_dpo_sequence_logps
T et_dpo_sequence_logps_raw
T et_dpo_validate_config
T et_dpo_weight_sha256
T et_dpo_weight_sha256_raw
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_dpo.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-dpo-exports"
diff -u "${scratch}/expected-dpo-exports" "${scratch}/actual-dpo-exports"

# Audit undefined imports on dpo_alignment.o (zero heap allocation, no threads/dlopen).
dpo_imports="$(nm -u "${scratch}/first/dpo_alignment.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    exp|fegetround|log|log1p|memcpy|memset) ;;
    *)
      printf 'unexpected DPO core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${dpo_imports}"

# Audit defined global exports on libeshkol_transformer_dpo_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_dpo_aot_bias_count
T et_dpo_aot_chosen_reward_mean
T et_dpo_aot_compute_ref_sha256
T et_dpo_aot_create
T et_dpo_aot_d_logit_chosen_bits
T et_dpo_aot_d_logit_rejected_bits
T et_dpo_aot_destroy
T et_dpo_aot_forward
T et_dpo_aot_forward_backward
T et_dpo_aot_get_d_logit_chosen
T et_dpo_aot_get_d_logit_rejected
T et_dpo_aot_get_d_policy_bias
T et_dpo_aot_get_d_policy_weight
T et_dpo_aot_get_policy_bias
T et_dpo_aot_get_policy_chosen_logit
T et_dpo_aot_get_policy_rejected_logit
T et_dpo_aot_get_policy_weight
T et_dpo_aot_head_backward
T et_dpo_aot_hidden_count
T et_dpo_aot_init_policy_from_ref
T et_dpo_aot_logit_count
T et_dpo_aot_mean_loss
T et_dpo_aot_poison_grads
T et_dpo_aot_preference_accuracy
T et_dpo_aot_project_policy_logits
T et_dpo_aot_ref_bias_bits
T et_dpo_aot_ref_sha256_word
T et_dpo_aot_ref_weight_bits
T et_dpo_aot_rejected_reward_mean
T et_dpo_aot_reward_margin_mean
T et_dpo_aot_set_chosen_mask
T et_dpo_aot_set_chosen_target
T et_dpo_aot_set_hidden_chosen
T et_dpo_aot_set_hidden_rejected
T et_dpo_aot_set_hyperparams
T et_dpo_aot_set_policy_bias
T et_dpo_aot_set_policy_chosen_logit
T et_dpo_aot_set_policy_rejected_logit
T et_dpo_aot_set_policy_weight
T et_dpo_aot_set_ref_bias
T et_dpo_aot_set_ref_chosen_logit
T et_dpo_aot_set_ref_rejected_logit
T et_dpo_aot_set_ref_weight
T et_dpo_aot_set_rejected_mask
T et_dpo_aot_set_rejected_target
T et_dpo_aot_token_count
T et_dpo_aot_weight_count
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_dpo_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on dpo_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/dpo_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|calloc|et_dpo_*|free|memcpy) ;;
    *)
      printf 'unexpected DPO bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_dpo.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_dpo_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_dpo.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_dpo_aot.a")"
printf 'DPO deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
