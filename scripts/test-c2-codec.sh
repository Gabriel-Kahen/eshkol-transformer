#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cc="${CC:-/usr/bin/clang}"; cxx="${CXX:-/usr/bin/clang++}"
[[ -x "${cc}" && -x "${cxx}" ]] || die "C2 codec gate requires clang and clang++"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-c2-codec.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all -fno-common -I "${PROJECT_ROOT}/native")
"${cc}" "${cflags[@]}" "${PROJECT_ROOT}/native/c2_checkpoint_codec.c" \
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c" \
  "${PROJECT_ROOT}/tests/c2/c2_codec_driver.c" -o "${temporary_dir}/driver"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/c2/c2_codec_header_cpp.cpp" -o "${temporary_dir}/header"
"${temporary_dir}/header"
"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/c2_checkpoint_codec.c" \
  -o "${temporary_dir}/codec.o"
nm -g --defined-only "${temporary_dir}/codec.o" | awk '{print $3}' | sort \
  > "${temporary_dir}/symbols.txt"
diff -u "${PROJECT_ROOT}/tests/c2/expected/checkpoint_codec_symbols.txt" \
  "${temporary_dir}/symbols.txt"
"${cc}" -std=c11 -I "${PROJECT_ROOT}/native" -MM \
  "${PROJECT_ROOT}/native/c2_checkpoint_codec.c" | \
  sed -e 's/^[^:]*://' -e 's/\\//g' | tr -s '[:space:]' '\n' | \
  sed "s#^${PROJECT_ROOT}/##" | rg '^native/' | sort -u \
  > "${temporary_dir}/source-closure.txt"
diff -u "${PROJECT_ROOT}/native/c2_checkpoint_codec_source_closure.txt" \
  "${temporary_dir}/source-closure.txt"
python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_codec.py" "${temporary_dir}/driver"
"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/native/c2_checkpoint_codec.c" \
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c" \
  "${PROJECT_ROOT}/tests/c2/c2_codec_driver.c" -o "${temporary_dir}/driver-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_codec.py" "${temporary_dir}/driver-san"
printf 'C2 CODEC PASS: exact bytes, parser round-trip, failure atomicity, C/C++ ABI, sanitizers\n'
