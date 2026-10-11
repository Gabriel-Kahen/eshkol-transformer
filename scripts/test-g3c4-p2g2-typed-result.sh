#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
out="${1:-$(mktemp -d)}"
mkdir -p "$out"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic
  -I "$root/include" -I "$root/native" -I "$root/src"
  -I "$root/src/eshkol_transformer"
  -DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_TESTING
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)
sources=("$root/tests/g3c4/test_p2_g2_typed_result.c"
  "$root/src/eshkol_transformer/m3_i64_integration.c"
  "$root/tests/m3t/kernel_fail_allocator.c"
  "$root/native/t1_i64_shell.c"
  "$root/native/n3k_primitives_provider.c"
  "$root/native/g3c4_primitives_provider.c"
  "$root/native/g3s_sampling_provider.c"
  "$root/native/g3n_primitives_provider.c"
  "$root/native/n2_primitives_provider.c"
  "$root/native/a2_kv_cache.c")
wraps=(-Wl,--wrap=et_kernel_runtime_dispatch
  -Wl,--wrap=et_i64_tensor_borrow_view_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_view_end_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1
  -Wl,--wrap=et_i64_tensor_copy_from_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1)
for mode in normal sanitize; do
  extra=(-O2)
  if [[ "$mode" == sanitize ]]; then
    extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "${extra[@]}" "${sources[@]}" \
    "${wraps[@]}" -lm -o "$out/$mode" \
    >"$out/$mode.compile.stdout" 2>"$out/$mode.compile.stderr"
  test ! -s "$out/$mode.compile.stderr"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 "$out/$mode" \
    >"$out/$mode.stdout" 2>"$out/$mode.stderr"
  test ! -s "$out/$mode.stderr"
done
"$out/normal" >"$out/repeat.stdout" 2>"$out/repeat.stderr"
test ! -s "$out/repeat.stderr"
cmp "$out/normal.stdout" "$out/repeat.stdout"
cmp "$out/normal.stdout" "$out/sanitize.stdout"
cat "$out/normal.stdout"
