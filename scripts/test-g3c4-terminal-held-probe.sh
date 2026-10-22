#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
evidence=${1:?usage: test-g3c4-terminal-held-probe.sh EVIDENCE_DIR [normal|sanitize]}
mode=${2:-normal}
flags=(-O2)
runtime=(env)
trace_flags=()
if [[ ${ET_G3C4_PROBE_TRACE:-0} == 1 ]]; then
  trace_flags=(-DET_G3C4_PROBE_TRACE -Wl,--wrap=exit -Wl,--wrap=abort -rdynamic
    -Wl,--wrap=et_g3c4_private_p2g2_terminal_ids_clone_release_v1)
fi
case "$mode" in
  normal) ;;
  sanitize)
    flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    runtime=(env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      UBSAN_OPTIONS=halt_on_error=1) ;;
  *) echo 'expected normal or sanitize' >&2; exit 2 ;;
esac
mkdir -p "$evidence"
sources=(src/eshkol_transformer/m3_i64_integration.c
  tests/m3t/kernel_fail_allocator.c native/t1_i64_shell.c
  native/n3k_primitives_provider.c native/g3c4_primitives_provider.c
  native/g3s_sampling_provider.c native/g3n_primitives_provider.c
  native/n2_primitives_provider.c native/a2_kv_cache.c)
cd "$PROJECT_ROOT"
"${CC:-$(lock_value supported_cc)}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -fstack-protector-all -ffp-contract=off -fexcess-precision=standard \
  -fno-fast-math -I include -I native -I src -I src/eshkol_transformer \
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE -DET_M3_TESTING \
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_TESTING \
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE \
  -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE "${flags[@]}" "${trace_flags[@]}" \
  tests/g3c4/test_terminal_held_probe.c "${sources[@]}" \
  -Wl,--wrap=et_kernel_runtime_dispatch \
  -Wl,--wrap=et_i64_tensor_borrow_view_v1 \
  -Wl,--wrap=et_a2_kv_cache_transaction_view_end_v1 \
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1 \
  -Wl,--wrap=et_i64_tensor_copy_from_v1 \
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1 -lm \
  -o "$evidence/probe-$mode" \
  >"$evidence/$mode.compile.stdout" 2>"$evidence/$mode.compile.stderr"
"${runtime[@]}" "$evidence/probe-$mode" \
  >"$evidence/$mode.stdout" 2>"$evidence/$mode.stderr"
test ! -s "$evidence/$mode.compile.stderr"
if [[ ${ET_G3C4_PROBE_TRACE:-0} != 1 ]]; then
  test ! -s "$evidence/$mode.stderr"
fi
grep -E '^G3-C4 held terminal/clone probe PASS: checks=[1-9][0-9]*$' \
  "$evidence/$mode.stdout"
