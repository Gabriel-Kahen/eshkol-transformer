#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
artifact="${root}/build/mp4"
mkdir -p "${artifact}"
scratch="$(mktemp -d "${artifact}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math \
  -fstack-protector-all -fPIC -I "${root}/include" \
  -c "${root}/native/mp4_mixed_precision_provider.c" \
  -o "${scratch}/mp4_mixed_precision_provider.o"
ar rcsD "${scratch}/libeshkol_transformer_mp4.a" "${scratch}/mp4_mixed_precision_provider.o"

exports="$(nm -g --defined-only "${scratch}/libeshkol_transformer_mp4.a" |
  awk 'NF >= 3 {print $(NF-1), $NF}' | LC_ALL=C sort)"
expected=$'T et_mp4_kernel_provider_v1\nT et_mp4_loss_scaler_init\nT et_mp4_loss_scaler_update'
if [[ "${exports}" != "${expected}" ]]; then
  printf 'unexpected MP4 exports:\n%s\n' "${exports}" >&2
  exit 1
fi

# Reject allocator, foreign-runtime and threading imports.
imports="$(nm -u "${scratch}/mp4_mixed_precision_provider.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|et_kernel_error_clear|\
    bcmp|fegetenv|fegetround|fesetenv|memcmp|memcpy|memset|strcmp|strlen) ;;
    *)
      printf 'unexpected MP4 dependency: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

mv -f "${scratch}/libeshkol_transformer_mp4.a" "${artifact}/"
printf 'MP4 deterministic archive and symbol audit: PASS (%s)\n' \
  "${artifact}/libeshkol_transformer_mp4.a"
