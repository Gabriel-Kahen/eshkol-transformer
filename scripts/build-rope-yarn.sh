#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

output="${1:-${root}/build/rope_yarn}"
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

# C++17 header check on public ABI and Eshkol bridge headers.
"${CXX}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${root}/include" -I "${root}/native" -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/rope_yarn_abi.h"
#include "rope_yarn_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_rope_yarn_config_t>);
static_assert(std::is_standard_layout_v<et_rope_yarn_invariance_result_t>);
static_assert(std::is_same_v<et_rope_yarn_status, int32_t>);
CPP

# Build both archives twice from fresh trial directories to prove determinism.
for trial in first second; do
  tdir="${scratch}/${trial}"
  mkdir -p -- "${tdir}"

  "${CC}" "${cflags[@]}" -c "${root}/native/rope_yarn.c" \
    -o "${tdir}/rope_yarn.o"
  ar rcsD "${tdir}/libeshkol_transformer_rope_yarn.a" \
    "${tdir}/rope_yarn.o"

  "${CC}" "${cflags[@]}" -c "${root}/native/rope_yarn_eshkol_bridge.c" \
    -o "${tdir}/rope_yarn_eshkol_bridge.o"
  ar rcsD "${tdir}/libeshkol_transformer_rope_yarn_aot.a" \
    "${tdir}/rope_yarn_eshkol_bridge.o"
done

cmp "${scratch}/first/libeshkol_transformer_rope_yarn.a" \
    "${scratch}/second/libeshkol_transformer_rope_yarn.a"
cmp "${scratch}/first/libeshkol_transformer_rope_yarn_aot.a" \
    "${scratch}/second/libeshkol_transformer_rope_yarn_aot.a"

# Audit defined global exports on libeshkol_transformer_rope_yarn.a.
cat > "${scratch}/expected-core-exports" <<'EOF'
T et_rope_yarn_backward_raw_v1
T et_rope_yarn_backward_v1
T et_rope_yarn_compute_freqs_raw_v1
T et_rope_yarn_compute_freqs_v1
T et_rope_yarn_forward_raw_v1
T et_rope_yarn_forward_v1
T et_rope_yarn_qk_dot_invariance_check_raw_v1
T et_rope_yarn_qk_dot_invariance_check_v1
T et_rope_yarn_validate_config_v1
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_rope_yarn.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-core-exports"
diff -u "${scratch}/expected-core-exports" "${scratch}/actual-core-exports"

# Audit undefined imports on rope_yarn.o (strictly zero allocator / IO / thread imports).
core_imports="$(nm -u "${scratch}/first/rope_yarn.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    cos|fegetenv|fegetround|fesetenv|floor|log|memcpy|memset|pow|sin|sincos) ;;
    *)
      printf 'unexpected RoPE YaRN core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit defined global exports on libeshkol_transformer_rope_yarn_aot.a.
cat > "${scratch}/expected-bridge-exports" <<'EOF'
T et_rope_yarn_aot_backward
T et_rope_yarn_aot_compute_freqs
T et_rope_yarn_aot_forward
T et_rope_yarn_aot_load_f32
T et_rope_yarn_aot_load_f64
T et_rope_yarn_aot_load_i64
T et_rope_yarn_aot_qk_invariance
T et_rope_yarn_aot_store_f32
T et_rope_yarn_aot_store_f64
T et_rope_yarn_aot_store_i64
EOF
nm -g --defined-only "${scratch}/first/libeshkol_transformer_rope_yarn_aot.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort \
  > "${scratch}/actual-bridge-exports"
diff -u "${scratch}/expected-bridge-exports" "${scratch}/actual-bridge-exports"

# Audit undefined imports on rope_yarn_eshkol_bridge.o (strictly zero allocator imports).
bridge_imports="$(nm -u "${scratch}/first/rope_yarn_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    et_rope_yarn_*|memcpy) ;;
    *)
      printf 'unexpected RoPE YaRN bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

mv -f -- "${scratch}/first/libeshkol_transformer_rope_yarn.a" "${output}/"
mv -f -- "${scratch}/first/libeshkol_transformer_rope_yarn_aot.a" "${output}/"

core_sha="$(sha256sum "${output}/libeshkol_transformer_rope_yarn.a")"
bridge_sha="$(sha256sum "${output}/libeshkol_transformer_rope_yarn_aot.a")"
printf 'RoPE YaRN deterministic archives and zero-alloc symbol audit: PASS (core=%s bridge=%s)\n' \
  "${core_sha%% *}" "${bridge_sha%% *}"
