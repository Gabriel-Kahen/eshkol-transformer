#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/o2_muon}"
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
#include "eshkol_transformer/o2_muon_abi.h"
#include "o2_muon_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_wsd_config>);
static_assert(std::is_standard_layout_v<et_muon_param_group>);
static_assert(std::is_standard_layout_v<et_muon_step_config>);
static_assert(std::is_same_v<et_muon_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to verify determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"
  "${CC}" "${cflags[@]}" -c "${root}/native/o2_muon_optimizer.c" \
    -o "${tdir}/o2_muon_optimizer.o"
  ar rcsD "${tdir}/libeshkol_transformer_o2_muon.a" \
    "${tdir}/o2_muon_optimizer.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/o2_muon_eshkol_bridge.c" \
    -o "${tdir}/o2_muon_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_o2_muon_aot.a" \
    "${tdir}/o2_muon_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_o2_muon.a" \
    "${scratch}/second/libeshkol_transformer_o2_muon.a"
cmp "${scratch}/first/libeshkol_transformer_o2_muon_aot.a" \
    "${scratch}/second/libeshkol_transformer_o2_muon_aot.a"

# Audit defined global exports on libeshkol_transformer_o2_muon.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_muon_hybrid_step
T et_muon_hybrid_step_wsd
T et_muon_newton_schulz5_f32
T et_muon_newton_schulz5_raw
T et_muon_orthogonality_error_f32
T et_muon_orthogonality_error_raw
T et_muon_validate_step_config
T et_wsd_lr
T et_wsd_validate_config
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_o2_muon.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on o2_muon_optimizer.o (no threading/dlopen/exec/IO).
core_imports="$(nm -u "${scratch}/first/o2_muon_optimizer.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    cos|fegetround|free|malloc|memcpy|memset|sqrt) ;;
    *)
      printf 'unexpected O2 Muon core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_o2_muon_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_muon_aot_add_adamw_group
T et_muon_aot_add_muon_group
T et_muon_aot_alloc_f32
T et_muon_aot_create
T et_muon_aot_destroy
T et_muon_aot_free_f32
T et_muon_aot_get_grad
T et_muon_aot_get_momentum
T et_muon_aot_get_param
T et_muon_aot_get_variance
T et_muon_aot_group_count
T et_muon_aot_group_ns5_ortho_error
T et_muon_aot_group_size
T et_muon_aot_last_lr
T et_muon_aot_last_ortho_error
T et_muon_aot_last_phase
T et_muon_aot_load_f32
T et_muon_aot_newton_schulz5
T et_muon_aot_orthogonality_error
T et_muon_aot_param_bits
T et_muon_aot_query_wsd
T et_muon_aot_set_grad
T et_muon_aot_set_param
T et_muon_aot_step_wsd
T et_muon_aot_store_f32
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_o2_muon_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on o2_muon_eshkol_bridge.o.
bridge_imports="$(nm -u "${scratch}/first/o2_muon_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|et_muon_*|et_wsd_*|free|malloc|memcpy) ;;
    *)
      printf 'unexpected O2 Muon bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_o2_muon.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_o2_muon_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_o2_muon.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_o2_muon_aot.a")"
printf 'O2 MUON deterministic archives and symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
