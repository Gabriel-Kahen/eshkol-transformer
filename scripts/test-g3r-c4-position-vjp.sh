#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?source root required}
out=${2:?fresh output directory required}
test ! -e "$out"
mkdir -p "$out"
cc=clang-21
"$cc" --version | head -1 | grep -F 'Ubuntu clang version 21.1.8'
"$cc" --version >"$out/toolchain.txt"
common=(-std=c11 -O1 -g -Wall -Wextra -Werror -fno-fast-math
        -ffp-contract=off -DET_M3T_C4_POSITION_VJP_PRIVATE
        -I"$source_root/include")
sources=("$source_root/tests/g3c4/test_position_vjp_private.c"
         "$source_root/native/m3t_c4_position_vjp_private.c"
         "$source_root/native/g3c4_primitives_provider.c"
         "$source_root/native/kernel_abi.c")
"$cc" "${common[@]}" "${sources[@]}" -lm -o "$out/normal"
"$out/normal" >"$out/normal.stdout" 2>"$out/normal.stderr"
"$out/normal" >"$out/repeat.stdout" 2>"$out/repeat.stderr"
cmp "$out/normal.stdout" "$out/repeat.stdout"
test ! -s "$out/normal.stderr"
test ! -s "$out/repeat.stderr"
"$cc" "${common[@]}" -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${sources[@]}" -lm -o "$out/sanitize"
ASAN_OPTIONS=detect_leaks=1 "$out/sanitize" \
  >"$out/sanitize.stdout" 2>"$out/sanitize.stderr"
cmp "$out/normal.stdout" "$out/sanitize.stdout"
test ! -s "$out/sanitize.stderr"
if "$cc" -std=c11 -I"$source_root/include" \
    -c "$source_root/native/m3t_c4_position_vjp_private.c" \
    -o "$out/feature-off.o" >"$out/feature-off.stdout" \
    2>"$out/feature-off.stderr"; then
  echo 'feature-off source compiled unexpectedly' >&2
  exit 1
fi
grep -F 'feature-private leaf' "$out/feature-off.stderr" >/dev/null
"$cc" -std=c11 -I"$source_root/include" \
  -c "$source_root/native/g3c4_primitives_provider.c" \
  -o "$out/ordinary-c4.o"
if nm -g --defined-only "$out/ordinary-c4.o" |
    grep -F 'et_m3t_c4_position_vjp_private_v1'; then
  echo 'ordinary C4 provider exports private VJP' >&2
  exit 1
fi
(
  cd "$source_root"
  sha256sum tests/g3c4/test_position_vjp_private.c \
    tests/g3c4/test_common.h \
    scripts/test-g3r-c4-position-vjp.sh \
    docs/g3/G3_R_C4_POSITION_VJP_LEAF.md \
    native/m3t_c4_position_vjp_private.c \
    native/m3t_c4_position_vjp_private.h \
    native/g3c4_primitives_provider.c native/kernel_abi.c \
    include/eshkol_transformer/g3c4_primitives_abi.h \
    include/eshkol_transformer/kernel_abi.h
) >"$out/source.sha256"
(
  cd "$out"
  sha256sum normal normal.stdout normal.stderr repeat.stdout repeat.stderr \
    sanitize sanitize.stdout sanitize.stderr feature-off.stdout \
    feature-off.stderr ordinary-c4.o source.sha256 toolchain.txt
) >"$out/SHA256SUMS"
echo 'PASS: pinned C4 position VJP normal/repeat/sanitizer/feature-off'
