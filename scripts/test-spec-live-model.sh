#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fexcess-precision=standard -fno-fast-math -fstack-protector-all
       -I include -I native -I src -I src/eshkol_transformer)
sources=(tests/spec_live/test_spec_live_model.c native/kernel_abi.c
         native/n3k_primitives_provider.c native/g3c4_primitives_provider.c
         native/g3s_sampling_provider.c native/g3n_primitives_provider.c
         native/n2_primitives_provider.c native/a2_kv_cache.c
         native/spec_paged_round.c native/spec_decode.c native/paged_kv.c)

"${cc}" "${flags[@]}" -O2 "${sources[@]}" -lm -o "${scratch}/witness"
"${scratch}/witness" > "${scratch}/normal.out"
"${scratch}/witness" > "${scratch}/repeat.out"
cmp "${scratch}/normal.out" "${scratch}/repeat.out"
"${cc}" "${flags[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" -lm -o "${scratch}/witness.san"
for run in san san-repeat; do
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "${scratch}/witness.san" > "${scratch}/${run}.out"
  cmp "${scratch}/normal.out" "${scratch}/${run}.out"
done
cat "${scratch}/normal.out"
printf 'SPEC5 live model normal/repeat/ASan/UBSan/LSan parity: PASS\n'
