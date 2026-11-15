#!/usr/bin/env bash
set -euo pipefail

# Build only the GGUF reader/writer library used by the IO5 development witness.
if [[ $# != 4 ]]; then
  echo "usage: $0 WORK_DIR C_COMPILER CXX_COMPILER VERSION" >&2
  exit 2
fi
work_dir=$1
cc=$2
cxx=$3
version=$4
commit=417a024b4fb6bc3116d41af7c067166a039bdf12
archive_sha256=eb2fb2d60331e78de996f4fd3f9cee955cfcab2546b7ad64100444730f39ede7

test ! -e "$work_dir"
"$cc" --version | head -1 | grep -F "version $version"
"$cxx" --version | head -1 | grep -F "version $version"
mkdir -p "$work_dir/source" "$work_dir/prefix/include" "$work_dir/prefix/lib"
curl -fsSL --retry 3 "https://codeload.github.com/ggml-org/ggml/tar.gz/$commit" \
  -o "$work_dir/ggml.tar.gz"
printf '%s  %s\n' "$archive_sha256" "$work_dir/ggml.tar.gz" | sha256sum -c -
tar -xzf "$work_dir/ggml.tar.gz" -C "$work_dir/source" --strip-components=1

CC="$cc" CXX="$cxx" cmake -S "$work_dir/source" -B "$work_dir/build" -G Ninja \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_CPU=OFF \
  -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_ALL_WARNINGS=OFF \
  -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "$work_dir/build" --target ggml-base --parallel 2
install -m 0644 "$work_dir/source/include/gguf.h" "$work_dir/source/include/ggml.h" \
  "$work_dir/prefix/include/"
install -m 0755 "$work_dir/build/src/libggml-base.so.1.0.0" "$work_dir/prefix/lib/"
ln -s libggml-base.so.1.0.0 "$work_dir/prefix/lib/libggml-base.so.1"
ln -s libggml-base.so.1 "$work_dir/prefix/lib/libggml-base.so"
printf 'IO5_NATIVE_SOURCE commit=%s archive_sha256=%s\n' "$commit" "$archive_sha256"
sha256sum "$work_dir/prefix/include/gguf.h" "$work_dir/prefix/include/ggml.h" \
  "$work_dir/prefix/lib/libggml-base.so.1.0.0"
