#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar cmp diff python3 timeout; do require_command "${command}"; done
cc="${CC:-/usr/bin/clang}"; cxx="${CXX:-/usr/bin/clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-d2-pair.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    local log
    shopt -s nullglob
    for log in "${tmp}"/*/*.stdout "${tmp}"/*/*.stderr; do
      if [[ -s "${log}" ]]; then
        printf 'C2 D2 diagnostic %s:\n' "${log}" >&2
        cat "${log}" >&2
      fi
    done
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

(cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 \
  python3 -m tests.c2.prepare_d2_cursor_pair_resources --output "${tmp}/resources-a" && \
  PYTHONDONTWRITEBYTECODE=1 \
  python3 -m tests.c2.prepare_d2_cursor_pair_resources --output "${tmp}/resources-b")
diff -ru "${tmp}/resources-a" "${tmp}/resources-b"

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
        -fno-common -fstack-protector-all -I "${PROJECT_ROOT}/include"
        -I "${PROJECT_ROOT}/native")
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${runtime}/i2_wave2_native_bridge.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" -o "${runtime}/p1_identity.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell f32_tensor; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/i64_tensor.c" \
  -o "${runtime}/i64_tensor.o"
"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/d2_native.c" \
  -o "${runtime}/d2_native.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_d2_pair.a" "${runtime}"/*.o

compile() {
  local label=$1
  mkdir -p "${tmp}/${label}"
  (cd "${tmp}/${label}" && env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${tmp}/cache-${label}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s 360s "${runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/internal/c1/lib" \
      -I "${PROJECT_ROOT}/internal/t2/lib" \
      -I "${PROJECT_ROOT}/internal/t1/lib" \
      -I "${PROJECT_ROOT}/internal/d2/lib" \
      -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" -L "${runtime}" \
      --lib eshkol_transformer_c2_d2_pair \
      "${PROJECT_ROOT}/tests/c2/c2_d2_cursor_pair_runtime.esk" \
      -o "${tmp}/${label}/cursor-pair" \
      >"${tmp}/${label}/compile.stdout" 2>"${tmp}/${label}/compile.stderr")
  test ! -s "${tmp}/${label}/compile.stderr"
}
compile a
compile b
for label in a b; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${tmp}/${label}/cursor-pair" \
    "${PROJECT_ROOT}/tests/t1/fixtures/byte_tokenizer_v1.tsv" \
    "${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
    "${tmp}/resources-a" >"${tmp}/${label}/run.stdout" \
    2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  grep -E '^C2 D2 CURSOR PAIR PASS: [0-9]+ checks$' \
    "${tmp}/${label}/run.stdout" >/dev/null
done
cmp "${tmp}/a/run.stdout" "${tmp}/b/run.stdout"
{
  cat "${PROJECT_ROOT}/native/d2_wave2_source_closure.txt"
  printf '%s\n' native/c2_d2_cursor_pair_extension.esk
} >"${tmp}/expected-closure"
cmp "${PROJECT_ROOT}/native/c2_d2_cursor_pair_source_closure.txt" \
  "${tmp}/expected-closure"
printf 'C2 D2 CURSOR PAIR PASS: actual D2 decode, external seek/next continuation, hostile mutation, regions, repeated strict AOT, source closure\n'
