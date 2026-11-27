#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${root}/build/spec_paged_round}"
archive=libeshkol_transformer_spec_paged_round.a

mkdir -p -- "${output}"
scratch="$(mktemp -d "${output}/.build.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

"${CC:-clang}" -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 \
  -fstack-protector-all -fPIC -I "${root}/include" \
  -c "${root}/native/spec_paged_round.c" -o "${scratch}/spec_paged_round.o"
ar rcsD "${scratch}/${archive}" "${scratch}/spec_paged_round.o"
exports="$(nm -g --defined-only "${scratch}/${archive}" | awk 'NF >= 3 {print $(NF-1), $NF}')"
if [[ "${exports}" != 'T et_spec_paged_round_greedy_v1' ]]; then
  printf 'unexpected SPEC_PAGED_ROUND exports:\n%s\n' "${exports}" >&2
  exit 1
fi
mv -f -- "${scratch}/${archive}" "${output}/"
printf 'SPEC_PAGED_ROUND archive and symbol audit: PASS (%s)\n' "${output}"
