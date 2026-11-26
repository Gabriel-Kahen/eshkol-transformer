#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${IO5_GGUF_NATIVE_PREFIX:?set IO5_GGUF_NATIVE_PREFIX to the pinned GGML prefix}"
archive="${IO5_GGML_SOURCE_ARCHIVE:?set IO5_GGML_SOURCE_ARCHIVE to the pinned GGML archive}"
cc="${CC:-clang}"
commit=417a024b4fb6bc3116d41af7c067166a039bdf12
archive_sha256=eb2fb2d60331e78de996f4fd3f9cee955cfcab2546b7ad64100444730f39ede7

printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c -
for header in ggml.h gguf.h; do
  cmp "${prefix}/include/${header}" \
    <(tar -xOf "$archive" "ggml-${commit}/include/${header}")
done
library="$(realpath -e "${prefix}/lib/libggml-base.so")"
sha256sum "${prefix}/include/ggml.h" "${prefix}/include/gguf.h" "$library"
"$cc" --version | head -1

scratch="$(mktemp -d "${TMPDIR:-/tmp}/io5-q8-ggml.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
"$cc" -std=c11 -O2 -Wall -Wextra -Werror -Wpedantic \
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math \
  -I "${root}/include" -I "${prefix}/include" \
  "${root}/tests/io5/q8_ggml_decoder_witness.c" \
  "${root}/native/io5_quant_codec.c" \
  -L "${prefix}/lib" -Wl,-rpath,"${prefix}/lib" -lggml-base \
  -o "${scratch}/q8-ggml-witness"
linked="$(env -u LD_PRELOAD -u LD_AUDIT LD_LIBRARY_PATH="${prefix}/lib" \
  ldd "${scratch}/q8-ggml-witness")"
grep -F "${prefix}/lib/libggml-base.so" <<< "$linked" >/dev/null
env -u LD_PRELOAD -u LD_AUDIT LD_LIBRARY_PATH="${prefix}/lib" \
  "${scratch}/q8-ggml-witness" \
  "${scratch}/raw-q8.gguf"
