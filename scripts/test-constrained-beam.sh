#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
scratch="$(mktemp -d)"
trap 'rm -rf -- "${scratch}"' EXIT
export LC_ALL=C
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
warnings=(-Wall -Wextra -Werror -Wpedantic)
common=(-std=c11 "${warnings[@]}" -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -I include)
deps=(native/beam_search.c native/constrained_decode.c)
test=tests/constrained_beam/test_constrained_beam_native.c

"${cxx}" -std=c++17 "${warnings[@]}" -I include -x c++ -fsyntax-only - <<'CPP'
#include "eshkol_transformer/constrained_beam_abi.h"
#include <type_traits>
using Step = et_beam_status (*)(et_beam_state_v1 *, const et_cd_vocab_index *,
                               const float *, size_t, size_t);
static_assert(std::is_same_v<decltype(&et_constrained_beam_step_v1), Step>);
CPP

"${cc}" "${common[@]}" -O2 -c native/constrained_beam.c -o "${scratch}/step.o"
exports="$(nm -g --defined-only "${scratch}/step.o" | awk '{print $(NF-1), $NF}')"
[[ "${exports}" == "T et_constrained_beam_step_v1" ]]
imports="$(nm -u "${scratch}/step.o" | awk '{print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|memcpy|    et_beam_get_active_beam_v1|et_beam_step_v1|et_cd_advance_token_v1|    et_cd_apply_logit_mask_v1|et_cd_vocab_index_info_get) ;;
    *) printf 'unexpected constrained beam dependency: %s\n' "${symbol}" >&2; exit 1 ;;
  esac
done <<< "${imports}"

"${cc}" "${common[@]}" -O2 "${test}" "${deps[@]}" "${scratch}/step.o" -lm \
  -o "${scratch}/step-test"
"${scratch}/step-test"
"${cc}" "${common[@]}" -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -fno-sanitize-recover=all "${test}" "${deps[@]}" \
  native/constrained_beam.c -lm -o "${scratch}/step-test-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "${scratch}/step-test-san"
