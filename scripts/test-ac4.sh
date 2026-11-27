#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
bash scripts/build-ac4.sh
bash scripts/build-mod5.sh

scratch="$(mktemp -d "${root}/build/ac4/.test.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
  -fexcess-precision=standard -fno-fast-math -I include)
archives=(build/ac4/libeshkol_ac4.a build/mod5/libeshkol_transformer_mod5.a)
"${cc}" "${cflags[@]}" -O2 -c native/kernel_abi.c -o "${scratch}/kernel.o"
"${cc}" "${cflags[@]}" -O2 tests/ac4/test_ac4_checkpointing.c "${archives[@]}" \
  "${scratch}/kernel.o" -lm -o "${scratch}/test"
"${scratch}/test"
"${scratch}/test" --benchmark

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I include \
  -x c++ - -x none "${archives[@]}" "${scratch}/kernel.o" -lm -o "${scratch}/header" <<'CPP'
#include "eshkol_transformer/ac4_checkpointing_abi.h"
int main() {
  et_ac4_config_v1 config{{16, 2, 3, 4, 2, 2}, ET_AC4_FORWARD_BACKWARD,
                               ET_AC4_POLICY_NONE, 0, 0.125f, 0.0f, 0, 0, 0};
  et_ac4_memory_v1 retained{}, replay{};
  if (et_ac4_plan_v1(&config, &retained) != ET_AC4_OK)
    return 1;
  config.policy = ET_AC4_POLICY_EVERY_K_REPLAY_GRADIENTS;
  config.checkpoint_every = config.shape.layers;
  return et_ac4_plan_v1(&config, &replay) == ET_AC4_OK &&
         replay.workspace_bytes < retained.workspace_bytes ? 0 : 1;
}
CPP
"${scratch}/header"
printf 'AC4 C++17 archive linkage: PASS\n'

sanitizers=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
export ASAN_OPTIONS="detect_leaks=${AC4_ASAN_DETECT_LEAKS:-0}:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
"${cc}" "${cflags[@]}" "${sanitizers[@]}" tests/ac4/test_ac4_checkpointing.c \
  native/ac4_checkpointing.c native/mod5_primitives_provider.c native/kernel_abi.c \
  -lm -o "${scratch}/test.san"
"${scratch}/test.san"
printf 'AC4 ASan/UBSan: PASS (LSan detect_leaks=%s)\n' "${AC4_ASAN_DETECT_LEAKS:-0}"
