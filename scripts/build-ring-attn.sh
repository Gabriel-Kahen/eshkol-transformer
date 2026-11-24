#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${1:-${root}/build/ring_attn}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
cflags=(-std=c11 "${warnings[@]}" -O2 -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math
        -fstack-protector-all -fPIC -I "${root}/include" -I "${root}/native")

"${CC}" "${cflags[@]}" -c "${root}/native/ring_attn.c" \
  -o "${scratch}/ring_attn.o"
ar rcsD "${scratch}/libeshkol_transformer_ring_attn.a" \
  "${scratch}/ring_attn.o"

"${CC}" "${cflags[@]}" -c "${root}/native/ring_attn_eshkol_bridge.c" \
  -o "${scratch}/ring_attn_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_ring_attn_aot.a" \
  "${scratch}/ring_attn_eshkol_bridge.o"

# Audit core ring attention exports & undefined imports.
core_exports="$(nm -g --defined-only "${scratch}/ring_attn.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_core_exports="$(printf 'T %s\n' \
  et_ring_attn_backward_v1 \
  et_ring_attn_bwd_block_step_v1 \
  et_ring_attn_comm_create_v1 \
  et_ring_attn_comm_destroy_v1 \
  et_ring_attn_dense_ref_backward_v1 \
  et_ring_attn_dense_ref_forward_v1 \
  et_ring_attn_forward_v1 \
  et_ring_attn_fwd_block_step_v1 \
  et_ring_attn_fwd_finalize_v1 \
  et_ring_attn_multiprocess_backward_v1 \
  et_ring_attn_multiprocess_forward_v1 \
  et_ring_attn_ring_exchange_v1 \
  et_ring_attn_validate_config_v1)"
if [[ "${core_exports}" != "${expected_core_exports}" ]]; then
  printf 'unexpected Ring Attention core exports:\n%s\n' "${core_exports}" >&2
  exit 1
fi

core_imports="$(nm -u "${scratch}/ring_attn.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|_exit|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    calloc|clock_gettime|close|exp|fcntl|fegetenv|fesetenv|fesetround|\
    fork|free|fstat|getpeername|getsockname|getsockopt|kill|log|malloc|\
    memcpy|memset|poll|recv|send|socketpair|sqrt|waitpid) ;;
    *)
      printf 'unexpected Ring Attention core dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# Audit Eshkol AOT bridge exports & undefined imports (zero allocator imports).
bridge_exports="$(nm -g --defined-only "${scratch}/ring_attn_eshkol_bridge.o" |
  awk 'NF {print $(NF-1), $NF}' | sort)"
expected_bridge_exports="$(printf 'T et_ring_attn_aot_%s\n' \
  config_bytes dense_backward dense_forward load_f32 load_f64 mp_backward \
  mp_forward set_config stats_bytes stats_bytes_received stats_bytes_sent \
  stats_executed_blocks stats_ring_rounds stats_skipped_causal \
  stats_total_blocks store_f32 store_f64)"
if [[ "${bridge_exports}" != "${expected_bridge_exports}" ]]; then
  printf 'unexpected Ring Attention AOT bridge exports:\n%s\n' "${bridge_exports}" >&2
  exit 1
fi

bridge_imports="$(nm -u "${scratch}/ring_attn_eshkol_bridge.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|\
    et_ring_attn_dense_ref_backward_v1|et_ring_attn_dense_ref_forward_v1|\
    et_ring_attn_multiprocess_backward_v1|et_ring_attn_multiprocess_forward_v1|\
    memcpy|memset) ;;
    *)
      printf 'unexpected Ring Attention AOT bridge dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"

# C++17 header syntax verification.
"${CXX}" -x c++ -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -fsyntax-only - <<'CPP'
#include "eshkol_transformer/ring_attn_abi.h"
#include "ring_attn_eshkol_bridge.h"
static_assert(ET_RING_ATTN_ABI_MAJOR == 1u);
static_assert(ET_RING_ATTN_OK == 0);
CPP

mv -f "${scratch}/libeshkol_transformer_ring_attn.a" "${artifact}/"
mv -f "${scratch}/libeshkol_transformer_ring_attn_aot.a" "${artifact}/"
printf 'RING_ATTN deterministic archives and symbol audit: PASS (%s/libeshkol_transformer_ring_attn.a)\n' "${artifact}"
