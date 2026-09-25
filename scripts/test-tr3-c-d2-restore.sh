#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar cmp diff nm python3 timeout; do require_command "${command}"; done
cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tr3-c-d2-restore.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    local log
    shopt -s nullglob
    for log in "${tmp}"/*/*.stdout "${tmp}"/*/*.stderr; do
      if [[ -s "${log}" ]]; then
        printf 'TR3-C D2 diagnostic %s:\n' "${log}" >&2
        sed -n '1,240p' "${log}" >&2
      fi
    done
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

(cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest -v tests.d2.test_tr3_c_d2_restore_scope)

{
  cat "${PROJECT_ROOT}/native/d2_wave2_source_closure.txt"
  printf '%s\n' native/tr3_c_d2_restore_extension.esk
} >"${tmp}/expected-closure"
cmp "${PROJECT_ROOT}/native/tr3_c_d2_restore_source_closure.txt" \
  "${tmp}/expected-closure"

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
        -fno-common -fstack-protector-all -I "${PROJECT_ROOT}/include"
        -I "${PROJECT_ROOT}/native")
"${cc}" "${cflags[@]}" -DET_TR3_C_D2_RESTORE \
  -DET_D2_NATIVE_TESTING -DET_I64_TENSOR_TESTING \
  "${PROJECT_ROOT}/tests/d2/test_tr3_c_d2_idle_probe.c" \
  "${PROJECT_ROOT}/native/d2_native.c" \
  "${PROJECT_ROOT}/native/i64_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  -o "${tmp}/test-tr3-c-d2-idle-probe"
for repetition in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${tmp}/test-tr3-c-d2-idle-probe" \
    >"${tmp}/native-${repetition}.stdout"
done
cmp "${tmp}/native-1.stdout" "${tmp}/native-2.stdout"
grep -F 'TR3-C D2 native idle probe PASS:' \
  "${tmp}/native-1.stdout" >/dev/null

"${cc}" "${cflags[@]}" -DET_TR3_C_D2_RESTORE \
  -DET_D2_NATIVE_TESTING -DET_I64_TENSOR_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/d2/test_tr3_c_d2_idle_probe.c" \
  "${PROJECT_ROOT}/native/d2_native.c" \
  "${PROJECT_ROOT}/native/i64_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  -o "${tmp}/test-tr3-c-d2-idle-probe-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${tmp}/test-tr3-c-d2-idle-probe-sanitized" \
  >"${tmp}/native-sanitized.stdout" \
  2>"${tmp}/native-sanitized.stderr"
test ! -s "${tmp}/native-sanitized.stderr"
cmp "${tmp}/native-1.stdout" "${tmp}/native-sanitized.stdout"

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
  -o "${tmp}/d2_native_ordinary.o"
nm -g --defined-only --format=posix "${tmp}/d2_native_ordinary.o" | \
  awk '{ print $1 }' >"${tmp}/ordinary-defined-symbols"
cmp "${PROJECT_ROOT}/native/d2_native_defined_symbols.txt" \
  "${tmp}/ordinary-defined-symbols"
if grep -E '^et_tr3_c_d2_' "${tmp}/ordinary-defined-symbols" >/dev/null; then
  die "ordinary D2 object exposed a TR3-C private symbol"
fi

"${cc}" "${cflags[@]}" -DET_TR3_C_D2_RESTORE \
  -c "${PROJECT_ROOT}/native/d2_native.c" \
  -o "${runtime}/d2_native.o"
nm -g --defined-only --format=posix "${runtime}/d2_native.o" | \
  awk '{ print $1 }' >"${tmp}/feature-defined-symbols"
grep -E '^et_tr3_c_d2_' "${tmp}/feature-defined-symbols" \
  >"${tmp}/feature-private-defined-symbols"
cmp "${PROJECT_ROOT}/native/tr3_c_d2_restore_defined_symbols.txt" \
  "${tmp}/feature-private-defined-symbols"
grep -Ev '^et_tr3_c_d2_' "${tmp}/feature-defined-symbols" \
  >"${tmp}/feature-base-defined-symbols"
cmp "${PROJECT_ROOT}/native/d2_native_defined_symbols.txt" \
  "${tmp}/feature-base-defined-symbols"
ar rcsD "${runtime}/libeshkol_transformer_tr3_c_d2_restore.a" \
  "${runtime}"/*.o

(cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 \
  python3 -m tests.c2.prepare_d2_cursor_pair_resources \
    --output "${tmp}/resources")

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
      --lib eshkol_transformer_tr3_c_d2_restore \
      "${PROJECT_ROOT}/tests/d2/tr3_c_d2_restore_runtime.esk" \
      -o "${tmp}/${label}/tr3-c-d2-restore" \
      >"${tmp}/${label}/compile.stdout" \
      2>"${tmp}/${label}/compile.stderr")
  test ! -s "${tmp}/${label}/compile.stderr"
}

compile a
compile b
for label in a b; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${tmp}/${label}/tr3-c-d2-restore" \
    "${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
    "${tmp}/resources" >"${tmp}/${label}/run.stdout" \
    2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  grep -E '^TR3-C D2 RESTORE PASS: [0-9]+ checks$' \
    "${tmp}/${label}/run.stdout" >/dev/null
done
cmp "${tmp}/a/run.stdout" "${tmp}/b/run.stdout"

printf 'TR3-C D2 RESTORE PASS: private native probe, source plan, witnesses, negatives, fixed commit tail, repeated strict AOT\n'
