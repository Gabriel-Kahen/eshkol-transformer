#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
for foundation in wave4-composed ac4 mod5 mp4 dist5 eval5; do
  bash "scripts/build-${foundation}.sh"
done

scratch="$(mktemp -d "${root}/build/wave4-composed/.test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
       -fno-fast-math -fexcess-precision=standard -I include)
sources=(tests/wave4/test_wave4_composed_trainer.c native/kernel_abi.c)
archives=(build/wave4-composed/libeshkol_wave4_composed.a build/ac4/libeshkol_ac4.a
          build/mod5/libeshkol_transformer_mod5.a build/mp4/libeshkol_transformer_mp4.a
          build/dist5/libeshkol_dist5.a build/eval5/libeshkol_eval5.a)
native=(native/wave4_composed_trainer.c native/ac4_checkpointing.c
        native/mod5_primitives_provider.c native/mp4_mixed_precision_provider.c
        native/dist5_collectives.c native/eval5_benchmark.c)
"${cc}" "${flags[@]}" -O2 "${sources[@]}" "${archives[@]}" -lm -o "${scratch}/test"
"${scratch}/test"

lsan="${WAVE4_COMPOSED_ASAN_DETECT_LEAKS:-0}"
export ASAN_OPTIONS="detect_leaks=${lsan}:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
"${cc}" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${sources[@]}" "${native[@]}" -lm -o "${scratch}/test-sanitized"
"${scratch}/test-sanitized"
printf 'Wave 4/5 composed ASan/UBSan: PASS (LSan detect_leaks=%s)\n' "${lsan}"
