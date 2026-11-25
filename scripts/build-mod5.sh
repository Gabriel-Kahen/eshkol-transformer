#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
artifact="${root}/build/mod5"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math \
  -fstack-protector-all -fPIC -I "${root}/include" \
  -c "${root}/native/mod5_primitives_provider.c" \
  -o "${scratch}/mod5_primitives_provider.o"
ar rcsD "${scratch}/libeshkol_transformer_mod5.a" "${scratch}/mod5_primitives_provider.o"

exports="$(nm -g --defined-only "${scratch}/mod5_primitives_provider.o" |
  awk 'NF {print $(NF-1), $NF}')"
if [[ "${exports}" != "T et_mod5_kernel_provider_v1" ]]; then
  printf 'unexpected MOD5 exports:\n%s\n' "${exports}" >&2
  exit 1
fi

# Explicit allowlist so new allocator, foreign-runtime or threading symbols fail.
imports="$(nm -u "${scratch}/mod5_primitives_provider.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|et_kernel_error_clear|\
    expf|fegetenv|fegetround|fesetenv|memcmp|memcpy|memset|snprintf|sqrtf|strcmp) ;;
    *)
      printf 'unexpected MOD5 dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

mv -f "${scratch}/libeshkol_transformer_mod5.a" "${artifact}/"
printf 'MOD5 archive and symbol audit: PASS (%s)\n' "${artifact}/libeshkol_transformer_mod5.a"
