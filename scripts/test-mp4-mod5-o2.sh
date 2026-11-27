#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
cc="${CC:-clang}"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off -fno-fast-math
       -fexcess-precision=standard -frounding-math -Iinclude -Inative)
sources=(tests/mp4/test_mp4_mod5_o2.c native/mod5_primitives_provider.c
         native/mp4_mixed_precision_provider.c native/kernel_abi.c
         native/f32_tensor.c native/o2_optimizer.c)
wrap=(-Wl,--wrap=et_f32_gradient_plan_prepare_v1 -Wl,--wrap=et_o2_optimizer_step_v1)
"$cc" "${flags[@]}" -O2 "${sources[@]}" "${wrap[@]}" -lm -o "$scratch/test"
"$cc" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${sources[@]}" "${wrap[@]}" -lm -o "$scratch/test.san"
for format in f16 bf16; do
  "$scratch/test" "$format" "$scratch/$format.bits" >"$scratch/$format.out"
  "$scratch/test" "$format" "$scratch/$format.repeat.bits" >"$scratch/$format.repeat.out"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$scratch/test.san" "$format" "$scratch/$format.san.bits" >"$scratch/$format.san.out"
  for extension in bits out; do
    cmp "$scratch/$format.$extension" "$scratch/$format.repeat.$extension"
    cmp "$scratch/$format.$extension" "$scratch/$format.san.$extension"
  done
  cat "$scratch/$format.out"
done
printf 'MP4/MOD5/O2 fresh-process normal/repeat/ASan/UBSan/LSan trace parity: PASS\n'
