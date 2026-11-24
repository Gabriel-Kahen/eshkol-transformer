#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/multi_lora_sgmv}"
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
#include "eshkol_transformer/multi_lora_sgmv_abi.h"
#include "multi_lora_sgmv_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_mlora_registry_config>);
static_assert(std::is_trivially_copyable_v<et_mlora_registry_config>);
static_assert(std::is_standard_layout_v<et_mlora_adapter_info>);
static_assert(std::is_trivially_copyable_v<et_mlora_adapter_info>);
static_assert(std::is_standard_layout_v<et_mlora_adamw_config>);
static_assert(std::is_trivially_copyable_v<et_mlora_adamw_config>);
static_assert(std::is_standard_layout_v<et_mlora_sgmv_stats>);
static_assert(std::is_trivially_copyable_v<et_mlora_sgmv_stats>);
static_assert(std::is_standard_layout_v<et_mlora_bundle_meta>);
static_assert(std::is_trivially_copyable_v<et_mlora_bundle_meta>);
static_assert(std::is_same_v<et_mlora_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/multi_lora_sgmv.c" \
    -o "${tdir}/multi_lora_sgmv.o"
  ar rcsD "${tdir}/libeshkol_transformer_multi_lora_sgmv.a" \
    "${tdir}/multi_lora_sgmv.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/multi_lora_sgmv_eshkol_bridge.c" \
    -o "${tdir}/multi_lora_sgmv_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_multi_lora_sgmv_aot.a" \
    "${tdir}/multi_lora_sgmv_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_multi_lora_sgmv.a" \
    "${scratch}/second/libeshkol_transformer_multi_lora_sgmv.a"
cmp "${scratch}/first/libeshkol_transformer_multi_lora_sgmv_aot.a" \
    "${scratch}/second/libeshkol_transformer_multi_lora_sgmv_aot.a"

# Audit defined global exports on libeshkol_transformer_multi_lora_sgmv.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_mlora_adamw_step_adapter_v1
T et_mlora_bundle_load_v1
T et_mlora_bundle_parse_v1
T et_mlora_bundle_save_v1
T et_mlora_bundle_serialize_v1
T et_mlora_bundle_size_v1
T et_mlora_get_adapter_grads_v1
T et_mlora_get_adapter_info_v1
T et_mlora_get_adapter_weights_v1
T et_mlora_get_base_sha256_v1
T et_mlora_list_adapters_v1
T et_mlora_pin_adapter_v1
T et_mlora_register_adapter_v1
T et_mlora_register_seeded_adapter_v1
T et_mlora_registry_config_get_v1
T et_mlora_registry_create_v1
T et_mlora_registry_destroy_v1
T et_mlora_set_adapter_weights_v1
T et_mlora_sgmv_backward_v1
T et_mlora_sgmv_forward_v1
T et_mlora_single_adapter_forward_v1
T et_mlora_unpin_adapter_v1
T et_mlora_unregister_adapter_v1
T et_mlora_validate_adamw_config_v1
T et_mlora_validate_registry_config_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_multi_lora_sgmv.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on multi_lora_sgmv.o (no threading/dlopen/exec).
core_imports="$(nm -u "${scratch}/first/multi_lora_sgmv.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|fchmod|fegetround|free|fstat|fsync|malloc|memcmp|memcpy|\
    memset|mkstemp|open|read|rename|sqrt|strrchr|unlink|write) ;;
    *)
      printf 'unexpected Multi-LoRA SGMV core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_multi_lora_sgmv_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_mlora_aot_adamw_step
T et_mlora_aot_adapter_generation
T et_mlora_aot_adapter_pin_count
T et_mlora_aot_adapter_rank
T et_mlora_aot_adapter_step
T et_mlora_aot_base_sha256_word
T et_mlora_aot_check_fp_env_trap
T et_mlora_aot_check_nonfinite_x
T et_mlora_aot_check_tampered_env
T et_mlora_aot_compute_base_sha256
T et_mlora_aot_compute_y_sha256
T et_mlora_aot_create
T et_mlora_aot_da_bits
T et_mlora_aot_db_bits
T et_mlora_aot_destroy
T et_mlora_aot_env_phase
T et_mlora_aot_finalize_base
T et_mlora_aot_get_a
T et_mlora_aot_get_b
T et_mlora_aot_get_da
T et_mlora_aot_get_db
T et_mlora_aot_get_dx
T et_mlora_aot_get_row_adapter
T et_mlora_aot_get_target
T et_mlora_aot_get_x
T et_mlora_aot_get_y
T et_mlora_aot_get_y_ref
T et_mlora_aot_last_active_segments
T et_mlora_aot_last_adapted_rows
T et_mlora_aot_last_base_rows
T et_mlora_aot_load_env_bundle
T et_mlora_aot_pin
T et_mlora_aot_register_seeded
T et_mlora_aot_save_env_bundle
T et_mlora_aot_set_a
T et_mlora_aot_set_b
T et_mlora_aot_set_bias
T et_mlora_aot_set_dy
T et_mlora_aot_set_row_adapter
T et_mlora_aot_set_target
T et_mlora_aot_set_w_base
T et_mlora_aot_set_x
T et_mlora_aot_sgmv_backward
T et_mlora_aot_sgmv_forward
T et_mlora_aot_single_row_ref_forward
T et_mlora_aot_unpin
T et_mlora_aot_unregister
T et_mlora_aot_w_count
T et_mlora_aot_x_count
T et_mlora_aot_y_bits
T et_mlora_aot_y_count
T et_mlora_aot_y_ref_bits
T et_mlora_aot_y_sha256_word
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_multi_lora_sgmv_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on multi_lora_sgmv_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/multi_lora_sgmv_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|et_mlora_*|fegetround|fesetround|free|getenv|malloc|memcpy|\
    memset|open|read|strcmp|symlink|unlink|write) ;;
    *)
      printf 'unexpected Multi-LoRA SGMV bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_multi_lora_sgmv.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_multi_lora_sgmv_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_multi_lora_sgmv.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_multi_lora_sgmv_aot.a")"
printf 'MULTI-LORA SGMV deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
