#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != /* || ! -x "$1" ]]; then
  echo 'usage: test-g3r-c2-stage-sha.sh ABSOLUTE_CLANG_COMPILER' >&2
  exit 2
fi
clang_cc="$1"
if [[ "$("$clang_cc" --version)" != *'clang version'* ]]; then
  echo 'native predecessor requires an explicit Clang compiler' >&2
  exit 2
fi
if ! command -v gcc >/dev/null 2>&1; then
  echo 'native predecessor requires GCC for independent parity' >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "${tmp}"' EXIT

python3 "${project_root}/tests/c2/prepare_checkpoint_load_fixtures.py" \
  "${tmp}/fixtures"
fixture="${tmp}/fixtures"
test "$(stat -c %s "${fixture}/larger-valid.c2")" = \
     "$(stat -c %s "${fixture}/exact-valid.c2")"
sha_a="$(sha256sum "${fixture}/valid.c2")"
sha_b="$(sha256sum "${fixture}/larger-valid.c2")"
sha_c="$(sha256sum "${fixture}/exact-valid.c2")"
sha_a="${sha_a%% *}"
sha_b="${sha_b%% *}"
sha_c="${sha_c%% *}"
test "${sha_b}" != "${sha_c}"
sources=(
  "${project_root}/native/c2_checkpoint_load_bridge.c"
  "${project_root}/native/c2_checkpoint_core.c"
  "${project_root}/native/c2_checkpoint_reader.c"
  "${project_root}/native/c2_checkpoint_format.c"
  "${project_root}/native/c2_x1_canonical.c"
  "${project_root}/native/checkpoint_io.c"
  "${project_root}/tests/c2/test_g3r_stage_sha.c"
)
cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -I "${project_root}/include"
  -I "${project_root}/native" -DET_G3R_C2_STAGE_SHA_PRIVATE
  -DET_C2_CHECKPOINT_LOAD_TESTING -DET_C2_CHECKPOINT_CORE_TESTING
  -DET_C2_CHECKPOINT_READER_TESTING -DET_CHECKPOINT_IO_TESTING
)
args=(
  "${fixture}/valid.c2" "${fixture}/larger-valid.c2"
  "${fixture}/exact-valid.c2" "${sha_a}" "${sha_c}"
  "${tmp}/mutable.c2"
)
"$clang_cc" "${cflags[@]}" "${sources[@]}" -o "${tmp}/clang"
"${tmp}/clang" "${args[@]}" >"${tmp}/clang.out"
gcc "${cflags[@]}" "${sources[@]}" -o "${tmp}/gcc"
"${tmp}/gcc" "${args[@]}" >"${tmp}/gcc.out"
cmp "${tmp}/clang.out" "${tmp}/gcc.out"
"$clang_cc" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${sources[@]}" -o "${tmp}/san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${tmp}/san" "${args[@]}" >"${tmp}/san.out"
cmp "${tmp}/clang.out" "${tmp}/san.out"

"$clang_cc" "${cflags[@]}" "${sources[@]:0:6}" \
  "${project_root}/tests/c2/test_checkpoint_load_bridge.c" \
  -o "${tmp}/old-bridge"
old_args=("${fixture}/valid.c2" "${fixture}/nonf32-buffer.c2"
  "${fixture}/nonf32-i64-buffer.c2" "${fixture}/larger-valid.c2"
  "${fixture}/exact-valid.c2")
"${tmp}/old-bridge" "${old_args[@]}" >"${tmp}/old-bridge.out"
rg -x 'C2 checkpoint load bridge PASS: 1163 checks' \
  "${tmp}/old-bridge.out" >/dev/null
off_flags=()
for flag in "${cflags[@]}"; do
  if [[ "${flag}" != -DET_G3R_C2_STAGE_SHA_PRIVATE ]]; then
    off_flags+=("${flag}")
  fi
done
"$clang_cc" "${off_flags[@]}" "${sources[@]:0:6}" \
  "${project_root}/tests/c2/test_checkpoint_load_bridge.c" \
  -o "${tmp}/old-bridge-off"
"${tmp}/old-bridge-off" "${old_args[@]}" >"${tmp}/old-bridge-off.out"
cmp "${tmp}/old-bridge.out" "${tmp}/old-bridge-off.out"

"$clang_cc" -std=c11 -I "${project_root}/native" -c \
  "${project_root}/native/c2_checkpoint_load_bridge.c" -o "${tmp}/off.o"
"$clang_cc" -std=c11 -I "${project_root}/native" \
  -DET_G3R_C2_STAGE_SHA_PRIVATE -c \
  "${project_root}/native/c2_checkpoint_load_bridge.c" -o "${tmp}/on.o"
nm -g --defined-only --format=posix "${tmp}/off.o" | \
  awk '$1 ~ /^et_/ { print $1 }' | LC_ALL=C sort >"${tmp}/off-symbols"
cmp "${project_root}/tests/c2/expected/checkpoint_load_bridge_symbols.txt" \
  "${tmp}/off-symbols"
nm --defined-only "${tmp}/on.o" | \
  rg 'et_c2_private_checkpoint_load_stage_with_sha_v1' >/dev/null
cat "${tmp}/clang.out"
