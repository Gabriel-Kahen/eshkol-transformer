#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
cc="${CC:-clang}"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include)
src=(tests/beam_cd/test_beam_cd_live_cpu.c native/beam_search.c
     native/constrained_beam.c native/constrained_decode.c native/t2_bpe_trainer.c
     native/ac4_checkpointing.c native/mod5_primitives_provider.c
     native/kernel_abi.c)

"${cc}" "${common[@]}" -O2 "${src[@]}" -lm -o "${scratch}/witness"
"${scratch}/witness" >"${scratch}/normal"
"${scratch}/witness" >"${scratch}/repeat"
"${cc}" "${common[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -fno-sanitize-recover=all "${src[@]}" -lm \
  -o "${scratch}/witness-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/witness-san" >"${scratch}/sanitized"
cmp "${scratch}/normal" "${scratch}/repeat"
cmp "${scratch}/normal" "${scratch}/sanitized"
cat "${scratch}/normal"
printf 'BEAM/CD live CPU optimized repeat and ASan/UBSan/LSan: PASS\n'
