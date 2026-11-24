#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export LC_ALL=C

artifact="${1:-${root}/build/spec_decode}"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
cflags=(-std=c11 "${warnings[@]}" -O2 -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -fstack-protector-all -fPIC)

# 1. Core speculative decoding library archive.
"${CC}" "${cflags[@]}" -I "${root}/include" \
  -c "${root}/native/spec_decode.c" -o "${scratch}/spec_decode.o"
ar rcsD "${scratch}/libeshkol_transformer_spec_decode.a" "${scratch}/spec_decode.o"
core_archive="${scratch}/libeshkol_transformer_spec_decode.a"

if [[ "$(ar t "${core_archive}")" != 'spec_decode.o' ]]; then
  printf 'unexpected spec_decode archive members\n' >&2
  exit 1
fi

core_exports="$(nm -g --defined-only "${core_archive}" |
  awk 'NF > 1 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected_core_exports="$(printf 'T %s\n' \
  et_spec_logits_to_probs_v1 \
  et_spec_sample_token_v1 \
  et_spec_verify_greedy_v1 \
  et_spec_verify_sample_v1)"
if [[ "${core_exports}" != "${expected_core_exports}" ]]; then
  printf 'unexpected spec_decode exports:\n%s\n' "${core_exports}" >&2
  exit 1
fi

core_imports="$(nm -u "${core_archive}" | awk 'NF > 1 {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|_GLOBAL_OFFSET_TABLE_|__stack_chk_fail|exp|log|memcpy|memset|fegetround) ;;
    *)
      printf 'unexpected spec_decode dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${core_imports}"

# 2. Eshkol AOT bridge archive.
"${CC}" "${cflags[@]}" -I "${root}/include" -I "${root}/native" \
  -c "${root}/native/spec_decode_eshkol_bridge.c" \
  -o "${scratch}/spec_decode_eshkol_bridge.o"
ar rcsD "${scratch}/libeshkol_transformer_spec_decode_aot.a" \
  "${scratch}/spec_decode_eshkol_bridge.o"
bridge_archive="${scratch}/libeshkol_transformer_spec_decode_aot.a"

if [[ "$(ar t "${bridge_archive}")" != 'spec_decode_eshkol_bridge.o' ]]; then
  printf 'unexpected spec_decode_aot archive members\n' >&2
  exit 1
fi

bridge_exports="$(nm -g --defined-only "${bridge_archive}" |
  awk 'NF > 1 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected_bridge_exports="$(printf 'T et_spec_aot_%s\n' \
  accepted_count create destroy emitted_count emitted_token eos_hit horizon \
  logits_to_probs next_counter prob rejected rng_draws sample_row \
  sampled_next_counter sampled_token set_draft_logit set_draft_token \
  set_target_logit verify_greedy verify_sample vocab_size)"
if [[ "${bridge_exports}" != "${expected_bridge_exports}" ]]; then
  printf 'unexpected spec_decode_aot exports:\n%s\n' "${bridge_exports}" >&2
  exit 1
fi

bridge_imports="$(nm -u "${bridge_archive}" | awk 'NF > 1 {print $NF}')"
uses_core=0
while IFS= read -r symbol; do
  case "${symbol}" in
    et_spec_logits_to_probs_v1|et_spec_sample_token_v1|et_spec_verify_greedy_v1|et_spec_verify_sample_v1)
      uses_core=$((uses_core + 1)) ;;
    ''|_GLOBAL_OFFSET_TABLE_|__stack_chk_fail|calloc|free) ;;
    *)
      printf 'unexpected spec_decode_aot dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${bridge_imports}"
if [[ "${uses_core}" != 4 ]]; then
  printf 'missing spec_decode_aot dependencies on core ABI (got %d/4)\n' "${uses_core}" >&2
  exit 1
fi

# 3. C++17 header syntax check.
"${CXX}" -x c++ -std=c++17 "${warnings[@]}" \
  -I "${root}/include" -I "${root}/native" -fsyntax-only - <<'CPP'
#include "eshkol_transformer/spec_decode_abi.h"
#include "spec_decode_eshkol_bridge.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_spec_sampling_params_v1>);
static_assert(std::is_standard_layout_v<et_spec_rng_state_v1>);
static_assert(std::is_standard_layout_v<et_spec_verify_result_v1>);
static_assert(ET_SPEC_AOT_ERR_NOMEM == -100);
CPP

mv -f "${core_archive}" "${bridge_archive}" "${artifact}/"
printf 'SPEC_DECODE C11/C++17 build and symbol audit: PASS (%s)\n' "${artifact}"
