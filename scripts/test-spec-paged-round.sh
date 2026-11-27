#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
common=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include)
test=tests/spec_paged/test_spec_paged_round.c
libs=("${scratch}/libeshkol_transformer_spec_paged_round.a"
      "${scratch}/libeshkol_transformer_spec_decode.a"
      "${scratch}/libeshkol_transformer_paged_kv.a")

bash scripts/build-spec-paged-round.sh "${scratch}"
bash scripts/build-spec-decode.sh "${scratch}"
bash scripts/build-paged-kv.sh "${scratch}"

"${cc}" "${common[@]}" -O2 "${test}" "${libs[@]}" -lm -o "${scratch}/round"
"${scratch}/round" > "${scratch}/normal.out"
"${cc}" "${common[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${test}" native/spec_paged_round.c \
  native/spec_decode.c native/paged_kv.c -lm -o "${scratch}/round.san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "${scratch}/round.san" > "${scratch}/san.out"
cmp "${scratch}/normal.out" "${scratch}/san.out"
cat "${scratch}/normal.out"

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I include \
  -x c++ -c - -o "${scratch}/header.o" <<'CPP'
#include "eshkol_transformer/spec_paged_round_abi.h"

int main() {
  et_spec_paged_round_result_v1 result{};
  auto status = et_spec_paged_round_greedy_v1(nullptr, &result);
  return status.kind == ET_SPEC_PAGED_ROUND_ERR_ARGUMENT ? 0 : 1;
}
CPP
"${cxx}" "${scratch}/header.o" "${libs[@]}" -lm -o "${scratch}/header"
"${scratch}/header"
printf 'SPEC_PAGED_ROUND C++17 linkage and optimized/ASan/UBSan/LSan parity: PASS\n'
