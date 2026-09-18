#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in cmp python3 rg timeout; do
  require_command "${command}"
done

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-/usr/bin/clang}" "${CXX:-/usr/bin/clang++}"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-c2-format.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fstack-protector-all -fno-common
  -I "${PROJECT_ROOT}/native"
)
sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
)

if rg -q 'x1_validate|x1_version_mismatch|x1_domain_mismatch|text_cursor|"required-features"' \
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"; then
  die "C2 format parser contains duplicate X1 canonical grammar"
fi

"${cc}" "${cflags[@]}" "${sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/c2_format_unit.c" \
  -o "${temporary_dir}/unit"
"${cc}" "${cflags[@]}" "${sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/c2_format_driver.c" \
  -o "${temporary_dir}/driver"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/c2/c2_format_header_cpp.cpp" \
  -o "${temporary_dir}/header-cpp"

"${temporary_dir}/unit"
"${temporary_dir}/header-cpp"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_format.py" \
  "${temporary_dir}/driver"

PYTHONHASHSEED=1 python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_format.py" \
  --emit "${temporary_dir}/fixture-a.c2"
PYTHONHASHSEED=987654 python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_format.py" \
  --emit "${temporary_dir}/fixture-b.c2"
cmp "${temporary_dir}/fixture-a.c2" "${temporary_dir}/fixture-b.c2"

"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/c2_format_unit.c" \
  -o "${temporary_dir}/unit-sanitized"
"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/c2_format_driver.c" \
  -o "${temporary_dir}/driver-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${temporary_dir}/unit-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 90s \
  python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_format.py" \
  "${temporary_dir}/driver-sanitized"

printf 'C2 FORMAT PASS: retained-byte parser, hostile inputs, deterministic fixture, C/C++ ABI, and sanitizers\n'
