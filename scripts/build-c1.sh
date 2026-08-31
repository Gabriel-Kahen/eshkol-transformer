#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar cmp nm; do
  require_command "${command}"
done
verify_toolchain

c1_build="$(project_build_dir)/c1"
c1_object="${c1_build}/checkpoint_io.o"
c1_archive="${c1_build}/libeshkol_transformer_checkpoint_io.a"
c1_cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"

mkdir -p "${c1_build}"
"${c1_cc}" \
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion \
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common \
  -fstack-protector-all -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/checkpoint_io.c" -o "${c1_object}.tmp"

LC_ALL=C nm -g --defined-only "${c1_object}.tmp" | \
  awk '$2 ~ /^[A-Z]$/ { print $3 }' | LC_ALL=C sort \
  >"${c1_build}/checkpoint_io_symbols.actual"
cmp "${PROJECT_ROOT}/native/checkpoint_io_symbols.txt" \
    "${c1_build}/checkpoint_io_symbols.actual"
if nm -a "${c1_object}.tmp" | grep -F 'checkpoint_io_test_' >/dev/null; then
  die "C1 production checkpoint I/O object contains a test hook"
fi

mv -f -- "${c1_object}.tmp" "${c1_object}"
rm -f -- "${c1_archive}.tmp"
ar rcsD "${c1_archive}.tmp" "${c1_object}"
mv -f -- "${c1_archive}.tmp" "${c1_archive}"
printf 'built C1 checkpoint I/O: %s\n' "${c1_archive}"
