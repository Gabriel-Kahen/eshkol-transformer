#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/qlora_nf4}"
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

# C++17 header syntax and standard-layout ABI contract check.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/qlora_nf4_abi.h"
#include "qlora_nf4_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_qlora_quant_spec>);
static_assert(std::is_standard_layout_v<et_qlora_quant_footprint>);
static_assert(std::is_standard_layout_v<et_qlora_linear_config>);
static_assert(std::is_standard_layout_v<et_qlora_adamw_config>);
static_assert(std::is_standard_layout_v<et_qlora_mlp_config>);
static_assert(std::is_standard_layout_v<et_qlora_checkpoint_meta>);
static_assert(std::is_same_v<et_qlora_status, int32_t>);
static_assert(std::is_same_v<et_qlora_quant_mode, uint32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/qlora_nf4.c" \
    -o "${tdir}/qlora_nf4.o"
  ar rcsD "${tdir}/libeshkol_transformer_qlora_nf4.a" "${tdir}/qlora_nf4.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/qlora_nf4_eshkol_bridge.c" \
    -o "${tdir}/qlora_nf4_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_qlora_nf4_aot.a" \
    "${tdir}/qlora_nf4_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_qlora_nf4.a" \
    "${scratch}/second/libeshkol_transformer_qlora_nf4.a"
cmp "${scratch}/first/libeshkol_transformer_qlora_nf4_aot.a" \
    "${scratch}/second/libeshkol_transformer_qlora_nf4_aot.a"

# Audit defined global exports on libeshkol_transformer_qlora_nf4.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
R ET_NF4_CODEBOOK
T et_qlora_adamw_step
T et_qlora_checkpoint_load
T et_qlora_checkpoint_parse
T et_qlora_checkpoint_save
T et_qlora_checkpoint_serialize
T et_qlora_checkpoint_size
T et_qlora_f32_linear_forward
T et_qlora_init_lora
T et_qlora_linear_backward
T et_qlora_linear_forward
T et_qlora_merge_to_f32
T et_qlora_mlp_base_sha256
T et_qlora_mlp_forward
T et_qlora_mlp_scratch_floats
T et_qlora_mlp_train_step
T et_qlora_nf4_dequantize
T et_qlora_nf4_quantize
T et_qlora_packed_footprint
T et_qlora_sha256_bytes
T et_qlora_validate_adamw_config
T et_qlora_validate_linear_config
T et_qlora_validate_mlp_config
T et_qlora_validate_quant_spec
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_qlora_nf4.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on qlora_nf4.o (no threading/dlopen/exec/stdio).
core_imports="$(nm -u "${scratch}/first/qlora_nf4.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    close|exp|fchmod|fegetround|floor|free|fstat|fsync|malloc|memcmp|memcpy|\
    memset|mkstemp|open|read|rename|sqrt|strrchr|unlink|write) ;;
    *)
      printf 'unexpected QLoRA NF4 core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_qlora_nf4_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_qlora_aot_adamw_step
T et_qlora_aot_base_sha256_word
T et_qlora_aot_check_negative_suite
T et_qlora_aot_compute_digests
T et_qlora_aot_configure_adamw
T et_qlora_aot_create
T et_qlora_aot_destroy
T et_qlora_aot_dq_bits_per_param
T et_qlora_aot_dq_compression_ratio
T et_qlora_aot_dq_packed_bytes
T et_qlora_aot_env_phase
T et_qlora_aot_eval_loss
T et_qlora_aot_f32_bytes
T et_qlora_aot_grad_check_count
T et_qlora_aot_grad_max_err
T et_qlora_aot_last_loss
T et_qlora_aot_load_env_checkpoint
T et_qlora_aot_max_merge_diff
T et_qlora_aot_negative_checks_passed
T et_qlora_aot_nf4_dq_rmse
T et_qlora_aot_nf4_sq_rmse
T et_qlora_aot_poison_trainable_state
T et_qlora_aot_prepare_model
T et_qlora_aot_run_grad_checks
T et_qlora_aot_save_env_checkpoint
T et_qlora_aot_sq_bits_per_param
T et_qlora_aot_sq_compression_ratio
T et_qlora_aot_sq_packed_bytes
T et_qlora_aot_state_sha256_word
T et_qlora_aot_train_step
T et_qlora_aot_uniform_int4_rmse
T et_qlora_aot_verify_merge_parity
T et_qlora_aot_y_sha256_word
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_qlora_nf4_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on qlora_nf4_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/qlora_nf4_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|cos|et_qlora_*|exp|floor|free|getenv|log|malloc|memcpy|\
    memset|open|read|sin|sincos|sqrt|strcmp|symlink|unlink|write) ;;
    *)
      printf 'unexpected QLoRA NF4 bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_qlora_nf4.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_qlora_nf4_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_qlora_nf4.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_qlora_nf4_aot.a")"
printf 'QLORA NF4 deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
