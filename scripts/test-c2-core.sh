#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in awk cmp gcc g++ nm python3 rg sed sort timeout tr; do
  require_command "${command}"
done

clang_cc="${CC:-/usr/bin/clang}"
clang_cxx="${CXX:-/usr/bin/clang++}"
[[ -x "${clang_cc}" && -x "${clang_cxx}" ]] || \
  die "C2 core gate requires clang and clang++"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-c2-core.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

for variant in base counter c1-header profile-exact profile-one profile-corrupt; do
  python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_format.py" \
    --emit "${temporary_dir}/${variant}.c2" --variant "${variant}"
done

sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/tests/c2/test_checkpoint_core.c"
)
cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
  -fno-common -I "${PROJECT_ROOT}/native"
  -DET_C2_CHECKPOINT_CORE_TESTING -DET_C2_CHECKPOINT_READER_TESTING
)
arguments=(
  "${temporary_dir}/base.c2"
  "${temporary_dir}/counter.c2"
  "${temporary_dir}/c1-header.c2"
  "${temporary_dir}/profile-exact.c2"
  "${temporary_dir}/profile-one.c2"
  "${temporary_dir}/profile-corrupt.c2"
)

for compiler in "${clang_cc}" /usr/bin/gcc; do
  name="$(basename "${compiler}")"
  "${compiler}" "${cflags[@]}" "${sources[@]}" \
    -o "${temporary_dir}/core-${name}"
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 60s \
      "${temporary_dir}/core-${name}" "${arguments[@]}" \
      >"${temporary_dir}/core-${name}-${run}.stdout"
  done
  cmp "${temporary_dir}/core-${name}-1.stdout" \
      "${temporary_dir}/core-${name}-2.stdout"
  rg -Fx 'C2 private checkpoint core: PASS (151 checks)' \
    "${temporary_dir}/core-${name}-1.stdout" >/dev/null
done

for compiler in "${clang_cxx}" /usr/bin/g++; do
  "${compiler}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
    -I "${PROJECT_ROOT}/native" \
    "${PROJECT_ROOT}/tests/c2/checkpoint_core_header_cpp.cpp" \
    -o "${temporary_dir}/header-$(basename "${compiler}")"
  "${temporary_dir}/header-$(basename "${compiler}")"
done

"${clang_cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/native" -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c" \
  -o "${temporary_dir}/core-production.o"
nm -g --defined-only "${temporary_dir}/core-production.o" | \
  awk '{print $3}' | sort >"${temporary_dir}/symbols.txt"
cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_core_symbols.txt" \
    "${temporary_dir}/symbols.txt"

"${clang_cc}" -std=c11 -I "${PROJECT_ROOT}/native" -MM \
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c" \
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c" \
  "${PROJECT_ROOT}/native/c2_x1_canonical.c" \
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c" | \
  sed -e 's/^[^:]*://' -e 's/\\//g' | tr -s '[:space:]' '\n' | \
  sed "s#^${PROJECT_ROOT}/##" | rg '^native/' | sort -u \
  >"${temporary_dir}/source-closure.txt"
cmp "${PROJECT_ROOT}/native/c2_checkpoint_core_source_closure.txt" \
    "${temporary_dir}/source-closure.txt"

"${clang_cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" \
  -o "${temporary_dir}/core-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${temporary_dir}/core-sanitized" "${arguments[@]}" >/dev/null

CC="${clang_cc}" CXX="${clang_cxx}" \
  /usr/bin/bash "${PROJECT_ROOT}/tests/c2/test_checkpoint_reader.sh" >/dev/null
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-c2-format.sh" >/dev/null

printf 'C2 CORE PASS: same-fd probes, retained validation, hostile mutation, C/C++, symbols, source closure, and sanitizers\n'
