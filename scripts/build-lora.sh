#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/lora}"
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
#include "eshkol_transformer/lora_adapter_abi.h"
#include "lora_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_lora_config>);
static_assert(std::is_standard_layout_v<et_lora_checkpoint_meta>);
static_assert(std::is_same_v<et_lora_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/lora_adapter.c" \
    -o "${tdir}/lora_adapter.o"
  ar rcsD "${tdir}/libeshkol_transformer_lora.a" "${tdir}/lora_adapter.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/lora_eshkol_bridge.c" \
    -o "${tdir}/lora_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_lora_aot.a" \
    "${tdir}/lora_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_lora.a" \
    "${scratch}/second/libeshkol_transformer_lora.a"
cmp "${scratch}/first/libeshkol_transformer_lora_aot.a" \
    "${scratch}/second/libeshkol_transformer_lora_aot.a"

# Audit defined global exports on libeshkol_transformer_lora.a.
cat > "${scratch}/expected-lora-exports" <<'EOF'
T et_lora_backward
T et_lora_backward_raw
T et_lora_base_forward
T et_lora_base_forward_raw
T et_lora_base_weight_sha256
T et_lora_base_weight_sha256_raw
T et_lora_checkpoint_load
T et_lora_checkpoint_load_raw
T et_lora_checkpoint_parse
T et_lora_checkpoint_save
T et_lora_checkpoint_save_raw
T et_lora_checkpoint_serialize
T et_lora_checkpoint_size
T et_lora_forward
T et_lora_forward_raw
T et_lora_init_adapter
T et_lora_init_adapter_raw
T et_lora_merge
T et_lora_merge_raw
T et_lora_unmerge
T et_lora_unmerge_raw
T et_lora_validate_config
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_lora.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-lora-exports"
diff -u "${scratch}/expected-lora-exports" "${scratch}/actual-lora-exports"

# Audit undefined imports on lora_adapter.o (no threading/dlopen/exec).
lora_imports="$(nm -u "${scratch}/first/lora_adapter.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    close|fegetround|free|fstat|fsync|malloc|memcmp|memcpy|memset|\
    mkstemp|open|read|rename|sqrtf|strrchr|unlink|write) ;;
    *)
      printf 'unexpected LoRA core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${lora_imports}"

# Audit defined global exports on libeshkol_transformer_lora_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_lora_aot_a_count
T et_lora_aot_b_count
T et_lora_aot_backward
T et_lora_aot_base_forward
T et_lora_aot_base_sha256_word
T et_lora_aot_check_tampered_env
T et_lora_aot_check_wrong_base_hash
T et_lora_aot_check_wrong_rank
T et_lora_aot_compute_base_sha256
T et_lora_aot_compute_y_sha256
T et_lora_aot_create
T et_lora_aot_destroy
T et_lora_aot_env_phase
T et_lora_aot_forward
T et_lora_aot_get_a
T et_lora_aot_get_b
T et_lora_aot_get_da
T et_lora_aot_get_db
T et_lora_aot_get_dx
T et_lora_aot_get_target
T et_lora_aot_get_w_base
T et_lora_aot_get_w_restored
T et_lora_aot_get_y
T et_lora_aot_get_y_base
T et_lora_aot_get_y_merged
T et_lora_aot_init_adapter
T et_lora_aot_load_env_checkpoint
T et_lora_aot_merge_and_forward
T et_lora_aot_save_env_checkpoint
T et_lora_aot_set_a
T et_lora_aot_set_b
T et_lora_aot_set_dy
T et_lora_aot_set_target
T et_lora_aot_set_w_base
T et_lora_aot_set_x
T et_lora_aot_unmerge
T et_lora_aot_w_base_bits
T et_lora_aot_w_count
T et_lora_aot_x_count
T et_lora_aot_y_base_bits
T et_lora_aot_y_bits
T et_lora_aot_y_count
T et_lora_aot_y_sha256_word
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_lora_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on lora_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/lora_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|close|et_lora_*|free|getenv|memcpy|open|read|strcmp|symlink|unlink|write) ;;
    *)
      printf 'unexpected LoRA bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_lora.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_lora_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_lora.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_lora_aot.a")"
printf 'LORA deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
