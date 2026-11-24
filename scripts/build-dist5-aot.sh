#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
artifact="${1:-${root}/build/dist5_aot}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
"${cc}" -std=c11 "${warnings[@]}" -O2 -ffp-contract=off \
  -fexcess-precision=standard -fno-fast-math -fstack-protector-all -fPIC \
  -I "${root}/include" -I "${root}/native" \
  -c "${root}/native/dist5_eshkol_bridge.c" \
  -o "${scratch}/dist5_eshkol_bridge.o"
archive="${scratch}/libeshkol_transformer_dist5_aot.a"
ar rcsD "${archive}" "${scratch}/dist5_eshkol_bridge.o"

if [[ "$(ar t "${archive}")" != 'dist5_eshkol_bridge.o' ]]; then
  printf 'unexpected DIST5 AOT archive members\n' >&2
  exit 1
fi

exports="$(nm -g --defined-only "${archive}" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected="$(cat <<'EOF'
T et_dist5_aot_allgather_f32
T et_dist5_aot_alloc_f32
T et_dist5_aot_allreduce_f32
T et_dist5_aot_barrier
T et_dist5_aot_broadcast_f32
T et_dist5_aot_close
T et_dist5_aot_equal_bits_f32
T et_dist5_aot_f32_add_inplace
T et_dist5_aot_f32_div_scalar_inplace
T et_dist5_aot_free_f32
T et_dist5_aot_load_f32
T et_dist5_aot_load_f32_bits
T et_dist5_aot_mod5_attention_backward
T et_dist5_aot_mod5_attention_forward
T et_dist5_aot_mod5_rms_backward
T et_dist5_aot_mod5_rms_forward
T et_dist5_aot_mod5_swiglu_backward
T et_dist5_aot_mod5_swiglu_forward
T et_dist5_aot_open
T et_dist5_aot_rank
T et_dist5_aot_read_byte
T et_dist5_aot_reduce_scatter_f32
T et_dist5_aot_sha256_hex_f32
T et_dist5_aot_store_f32
T et_dist5_aot_store_f32_bits
T et_dist5_aot_sync_gradients_f32
T et_dist5_aot_world_size
EOF
)"
if [[ "${exports}" != "${expected}" ]]; then
  printf 'unexpected DIST5 AOT bridge exports:\n%s\n' "${exports}" >&2
  exit 1
fi

imports="$(nm -u "${scratch}/dist5_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    accept4|bcmp|bind|calloc|clock_gettime|close|connect|\
    et_dist5_allgather|et_dist5_allreduce|et_dist5_barrier|\
    et_dist5_broadcast|et_dist5_comm_create|et_dist5_comm_destroy|\
    et_dist5_reduce_scatter|et_dist5_sync_gradients_f32|\
    et_kernel_runtime_destroy|et_kernel_runtime_discover|\
    et_kernel_runtime_dispatch|et_mod5_kernel_provider_v1|\
    fegetenv|fesetenv|free|getsockopt|listen|memcmp|memcpy|memset|\
    nanosleep|poll|recv|send|snprintf|socket|stat|strcmp|unlink) ;;
    *)
      printf 'unexpected DIST5 AOT bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

"${cxx}" -x c++ -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -fsyntax-only - <<'CPP'
#include "dist5_eshkol_bridge.h"
static_assert(ET_DIST5_OK == 0);
CPP

mv -f "${archive}" "${artifact}/"
printf 'DIST5 AOT deterministic archive and symbol audit: PASS (%s)\n' \
  "${artifact}/libeshkol_transformer_dist5_aot.a"
