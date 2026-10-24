#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evidence="${1:-${project_root}/build/cpu4-p3-resume-native}"
mkdir -p -- "$evidence"
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -O1 -g
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "$project_root/include" -I "$project_root/native"
  -I "$project_root/src" -I "$project_root/src/eshkol_transformer"
  -DET_I64_TENSOR_TESTING -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE
  -DET_F32_TENSOR_STORAGE_QUERY_PRIVATE -DET_A2_KV_CACHE_TESTING
  -DET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
  -DET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE)
sources=("$project_root/tests/g3c4/test_cpu4_p3_resume_native.c"
  "$project_root/src/eshkol_transformer/m3_i64_integration.c"
  "$project_root/tests/m3t/kernel_fail_allocator.c"
  "$project_root/native/t1_i64_shell.c"
  "$project_root/native/n3k_primitives_provider.c"
  "$project_root/native/g3c4_primitives_provider.c"
  "$project_root/native/g3s_sampling_provider.c"
  "$project_root/native/g3n_primitives_provider.c"
  "$project_root/native/n2_primitives_provider.c"
  "$project_root/native/a2_kv_cache.c")
wraps=(-Wl,--wrap=et_kernel_runtime_dispatch
  -Wl,--wrap=et_kernel_runtime_discover
  -Wl,--wrap=et_a2_kv_cache_create_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_begin_v1
  -Wl,--wrap=et_i64_tensor_borrow_view_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_view_end_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_abort_v1
  -Wl,--wrap=et_i64_tensor_copy_from_v1
  -Wl,--wrap=et_a2_kv_cache_transaction_commit_v1)
"$cc" "${flags[@]}" "${sources[@]}" "${wraps[@]}" -lm \
  -o "$evidence/test" >"$evidence/compile.stdout" \
  2>"$evidence/compile.stderr"
"$evidence/test" >"$evidence/runtime.stdout" \
  2>"$evidence/runtime.stderr"
grep -Eq '^CPU4 C4 P3 resume native PASS: checks=[1-9][0-9]*$' \
  "$evidence/runtime.stdout"
