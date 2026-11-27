#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -fno-fast-math -I include -I native
       -I src -I src/eshkol_transformer)
sources=(tests/paged_kv/test_model_attention.c tests/m3t/kernel_fail_allocator.c
         native/n3k_primitives_provider.c native/g3c4_primitives_provider.c
         native/g3s_sampling_provider.c native/a2_kv_cache.c native/paged_kv.c)

"${cc}" "${flags[@]}" -O2 "${sources[@]}" \
  -Wl,--wrap=et_kernel_runtime_dispatch -lm -o "${scratch}/witness"
"${scratch}/witness" >"${scratch}/normal.out"
"${cc}" "${flags[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" \
  -Wl,--wrap=et_kernel_runtime_dispatch -lm -o "${scratch}/witness.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/witness.san" >"${scratch}/san.out"
cmp "${scratch}/normal.out" "${scratch}/san.out"
cat "${scratch}/normal.out"
printf 'PKV5 model attention normal/ASan/UBSan/LSan parity: PASS\n'
