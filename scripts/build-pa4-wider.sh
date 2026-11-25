#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-clang}"
out="${root}/build/pa4_wider"
mkdir -p "${out}"
scratch="$(mktemp -d "${out}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
cd "${root}"

"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -ffp-contract=off \
  -fPIC -I include -c native/pa4_wider_attention_provider.c \
  -o "${scratch}/pa4_wider_attention_provider.o"

exports="$(nm -g --defined-only "${scratch}/pa4_wider_attention_provider.o" |
  awk 'NF {print $(NF-1), $NF}' | LC_ALL=C sort)"
if [[ "${exports}" != $'T et_pa4_wider_online_provider_v1\nT et_pa4_wider_ref_provider_v1' ]]; then
  printf 'unexpected PA4-Wider exports:\n%s\n' "${exports}" >&2
  exit 1
fi

imports="$(nm -u "${scratch}/pa4_wider_attention_provider.o" | awk 'NF {print $NF}')"
while IFS= read -r symbol; do
  case "${symbol}" in
    ''|__errno_location|__stack_chk_fail|_GLOBAL_OFFSET_TABLE_|expf|fegetenv|fegetround|fesetenv|\
    memcpy|memset|sqrtf|strcmp|strlen) ;;
    *)
      printf 'unexpected PA4-Wider import: %s\n' "${symbol}" >&2
      exit 1
      ;;
  esac
done <<< "${imports}"

archive=libeshkol_transformer_pa4_wider.a
ar rcsD "${scratch}/${archive}" "${scratch}/pa4_wider_attention_provider.o"
mv -f "${scratch}/${archive}" "${out}/"
printf 'PA4-Wider archive and symbol audit: PASS\n'
