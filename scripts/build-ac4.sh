#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
artifact="${root}/build/ac4"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

warnings=(-Wall -Wextra -Werror -Wpedantic)
"${cc}" -std=c11 "${warnings[@]}" -O2 -ffp-contract=off -fexcess-precision=standard \
  -fno-fast-math -fstack-protector-all -fPIC -I "${root}/include" \
  -c "${root}/native/ac4_checkpointing.c" -o "${scratch}/ac4_checkpointing.o"
ar rcsD "${scratch}/libeshkol_ac4.a" "${scratch}/ac4_checkpointing.o"
archive="${scratch}/libeshkol_ac4.a"
if [[ "$(ar t "${archive}")" != 'ac4_checkpointing.o' ]]; then
  printf 'unexpected AC4 archive members\n' >&2
  exit 1
fi

exports="$(nm -g --defined-only "${archive}" |
  awk 'NF > 1 {print $(NF-1), $NF}' | LC_ALL=C sort)"
if [[ "${exports}" != $'T et_ac4_plan_v1\nT et_ac4_run_v1' ]]; then
  printf 'unexpected AC4 exports:\n%s\n' "${exports}" >&2
  exit 1
fi

imports="$(nm -u "${archive}" | awk 'NF > 1 {print $NF}')"
uses_mod5=0
while IFS= read -r symbol; do
  case "${symbol}" in
    et_mod5_kernel_provider_v1) uses_mod5=1 ;;
    ''|_GLOBAL_OFFSET_TABLE_|__stack_chk_fail|memcpy|memset) ;;
    *)
      printf 'unexpected AC4 dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"
if [[ "${uses_mod5}" != 1 ]]; then
  printf 'missing AC4 dependency on MOD5\n' >&2
  exit 1
fi

"${cxx}" -x c++ -std=c++17 "${warnings[@]}" -I "${root}/include" -fsyntax-only - <<'CPP'
#include "eshkol_transformer/ac4_checkpointing_abi.h"
#include <type_traits>
static_assert(std::is_standard_layout_v<et_ac4_config_v1>);
static_assert(std::is_standard_layout_v<et_ac4_io_v1>);
static_assert(std::is_standard_layout_v<et_ac4_result_v1>);
CPP

mv -f "${archive}" "${artifact}/"
printf 'AC4 C11/C++17 build and symbol audit: PASS (%s)\n' "${artifact}/libeshkol_ac4.a"
