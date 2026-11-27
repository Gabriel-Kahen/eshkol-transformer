#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
bash scripts/build-mod5.sh

scratch="$(mktemp -d "${root}/build/mod5/.o2-continuation.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fno-fast-math -fexcess-precision=standard -frounding-math
       -I include -I native -DET_O2_TESTING)
source=(tests/mod5/test_mod5_o2_continuation.c native/kernel_abi.c
        native/f32_tensor.c native/o2_optimizer.c)
"${cc}" "${flags[@]}" -O2 "${source[@]}" \
  build/mod5/libeshkol_transformer_mod5.a -lm -o "${scratch}/test"
"${scratch}/test"

sanitizers=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
"${cc}" "${flags[@]}" "${sanitizers[@]}" "${source[@]}" \
  native/mod5_primitives_provider.c -lm -o "${scratch}/test-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "${scratch}/test-sanitized"
printf 'MOD5/O2 ASan/UBSan/LSan: PASS\n'
