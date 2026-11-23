#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
python3 -m unittest -v tests.ci.test_attention_predecessor_pins
cc="${CC:-clang}"
out="${PA4_BUILD_DIR:-${root}/build/pa4}"
mkdir -p "${out}/baseline"
baseline="${out}/baseline/a2_attention_provider.c"
git show ee24ef461ef7e313c36c6c6f97ba4f3d8155bac7:native/a2_attention_provider.c \
  >"${baseline}"
printf '%s  %s\n' \
  fdb4ca172ca7aa282e787ff42b2998592c8b77a30f67d35680f7e38b2f550b02 \
  "${baseline}" | sha256sum --check --status

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -ffp-contract=off -O2 -I "${root}/include")
"${cc}" "${flags[@]}" -c "${baseline}" -o "${out}/baseline.o"
"${cc}" "${flags[@]}" -c native/a2_attention_provider.c -o "${out}/default.o"
cmp "${out}/baseline.o" "${out}/default.o"
"${cc}" "${flags[@]}" -DET_PA4_ATTENTION_CANDIDATE \
  -c native/a2_attention_provider.c -o "${out}/candidate.o"
"${cc}" "${flags[@]}" -DET_PA4_TEST tests/a2/test_attention_provider.c \
  "${out}/candidate.o" native/kernel_abi.c -lm -o "${out}/test-provider"
"${out}/test-provider"
"${cc}" "${flags[@]}" tests/pa4/test_candidate_parity.c \
  "${out}/baseline.o" "${out}/candidate.o" native/kernel_abi.c -lm \
  -o "${out}/test-parity"
"${out}/test-parity"
nm -g --defined-only "${out}/candidate.o" | \
  awk '$2 == "T" { print $3 }' | \
  cmp - <(printf '%s\n' et_pa4_kernel_provider_v1)
nm -u "${out}/candidate.o" | \
  awk '$NF ~ /^(malloc|calloc|realloc|free|aligned_alloc|posix_memalign)$/ { print }' | \
  cmp - /dev/null
