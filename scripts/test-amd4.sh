#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-cc}"
cxx="${CXX:-c++}"
work="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-amd4.XXXXXX")"
trap 'rm -rf -- "${work}"' EXIT
lib=libeshkol_transformer_amd4.a
warnings=(-Wall -Wextra -Werror -Wpedantic -I "${root}/include")
sanitize=(-fsanitize=address,undefined -fno-omit-frame-pointer)
test_source="${root}/tests/amd4/test_amd4_observability.c"
k1_source="${root}/native/kernel_abi.c"

export LC_ALL=C
export ASAN_OPTIONS="detect_leaks=${AMD4_ASAN_DETECT_LEAKS:-0}:halt_on_error=1"
export UBSAN_OPTIONS=halt_on_error=1

bash "${root}/scripts/build-amd4.sh" "${work}/first"
bash "${root}/scripts/build-amd4.sh" "${work}/second"
cmp -- "${work}/first/${lib}" "${work}/second/${lib}"
(cd -- "${work}/first" && sha256sum -c "${lib}.sha256")

"${cc}" -std=c11 "${warnings[@]}" "${test_source}" "${k1_source}" \
  "${work}/first/${lib}" -ldl -o "${work}/test-amd4"
"${work}/test-amd4"
"${cxx}" -std=c++17 "${warnings[@]}" "${root}/tests/amd4/test_amd4_header_cpp.cpp" \
  "${work}/first/${lib}" -ldl -o "${work}/header-cpp"
"${work}/header-cpp"
printf 'AMD4 C++17 linkage passed\n'

"${cc}" -std=c11 "${warnings[@]}" "${sanitize[@]}" -g \
  "${test_source}" "${k1_source}" "${root}/native/amd4_hip_observability.c" \
  -ldl -o "${work}/test-amd4-sanitized"
"${work}/test-amd4-sanitized"
printf 'AMD4 archive reproducibility, ASan and UBSan passed (detect_leaks=%s)\n' \
  "${AMD4_ASAN_DETECT_LEAKS:-0}"
