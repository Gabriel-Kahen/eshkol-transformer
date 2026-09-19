#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
build_dir=$(mktemp -d /tmp/et-c2-reader-test-XXXXXX)
trap 'rm -rf -- "$build_dir"' EXIT

c_compiler=${CC:-clang}
cxx_compiler=${CXX:-clang++}
common=(-Wall -Wextra -Werror -pedantic -I"$repo_root/native")

"$c_compiler" -std=c11 "${common[@]}" \
  -DET_C2_CHECKPOINT_READER_TESTING \
  "$repo_root/native/c2_checkpoint_reader.c" \
  "$repo_root/tests/c2/test_checkpoint_reader.c" \
  -o "$build_dir/checkpoint-reader-test"
"$build_dir/checkpoint-reader-test"

"$cxx_compiler" -std=c++17 "${common[@]}" \
  "$repo_root/tests/c2/checkpoint_reader_header_cpp.cpp" \
  -o "$build_dir/checkpoint-reader-header-cpp"
"$build_dir/checkpoint-reader-header-cpp"

"$c_compiler" -std=c11 "${common[@]}" -c \
  "$repo_root/native/c2_checkpoint_reader.c" \
  -o "$build_dir/c2_checkpoint_reader.o"
nm -g --defined-only "$build_dir/c2_checkpoint_reader.o" | \
  awk '{print $3}' | sort >"$build_dir/actual-symbols.txt"
diff -u "$repo_root/tests/c2/expected/checkpoint_reader_symbols.txt" \
  "$build_dir/actual-symbols.txt"

"$c_compiler" -std=c11 "${common[@]}" \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DET_C2_CHECKPOINT_READER_TESTING \
  "$repo_root/native/c2_checkpoint_reader.c" \
  "$repo_root/tests/c2/test_checkpoint_reader.c" \
  -o "$build_dir/checkpoint-reader-sanitize"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  "$build_dir/checkpoint-reader-sanitize"
