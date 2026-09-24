#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in clang cmp gcc grep nm python3 readelf timeout; do
  require_command "${command}"
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tr3-c-bindings.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    local log
    shopt -s nullglob
    for log in "${tmp}"/*.stdout "${tmp}"/*.stderr; do
      if [[ -s "${log}" ]]; then
        printf 'TR3-C binding diagnostic %s:\n' "${log}" >&2
        sed -n '1,240p' "${log}" >&2
      fi
    done
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.tr3_restore_bindings.test_source_contract

python3 - native/tr3_c_restore_bindings_extension.esk <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
stack = []
line = column = 1
index = 0
in_string = False
escaped = False
while index < len(text):
    char = text[index]
    if char == "\n":
        line += 1
        column = 0
    if in_string:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = False
    elif char == '"':
        in_string = True
    elif char == ";":
        end = text.find("\n", index)
        index = len(text) if end < 0 else end - 1
    elif char == "(":
        stack.append((line, column))
    elif char == ")":
        if not stack:
            raise SystemExit(f"{path}:{line}:{column}: unexpected close")
        stack.pop()
    index += 1
    column += 1
if in_string or stack:
    raise SystemExit(f"{path}: incomplete reader form")
print(f"TR3-C binding reader preflight PASS: {path}")
PY

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -O0
        -ffp-contract=off -fexcess-precision=standard -frounding-math
        -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
        -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
feature=(-DET_TR3_C_RESTORE_BINDINGS -DET_I2_PRIVATE_OWNED_CLONE_MATCH
         -DET_TR3_C_I2_RESTORE_PRIVATE -DET_TR3_C_O2_RESTORE_NATIVE)
test_feature=(-DET_I2_NATIVE_HELPERS_ONLY -DET_TR3_O2_STEP_CLEAR_NATIVE
              -DET_F32_TENSOR_TESTING -DET_O2_TESTING)

symbols() {
  local mode=$1 object=$2 output=$3
  nm -g "--${mode}-only" --format=posix "${object}" |
    awk '{ print $1 }' |
    grep -v '^_GLOBAL_OFFSET_TABLE_$' |
    LC_ALL=C sort >"${output}"
}

for compiler in clang gcc; do
  object="${tmp}/${compiler}-bindings.o"
  if "${compiler}" "${cflags[@]}" -c \
      "${PROJECT_ROOT}/native/tr3_c_restore_bindings.c" \
      -o "${tmp}/${compiler}-ungated.o" \
      >"${tmp}/${compiler}-ungated.stdout" \
      2>"${tmp}/${compiler}-ungated.stderr"; then
    die "${compiler} accepted ungated TR3-C bindings"
  fi
  grep -F 'require the private binding feature' \
    "${tmp}/${compiler}-ungated.stderr" >/dev/null
  "${compiler}" "${cflags[@]}" "${feature[@]}" -c \
    "${PROJECT_ROOT}/native/tr3_c_restore_bindings.c" -o "${object}"
  symbols defined "${object}" "${tmp}/${compiler}.defined"
  symbols undefined "${object}" "${tmp}/${compiler}.undefined"
  cmp "${PROJECT_ROOT}/native/tr3_c_restore_bindings_defined_symbols.txt" \
    "${tmp}/${compiler}.defined"
  cmp "${PROJECT_ROOT}/native/tr3_c_restore_bindings_undefined_symbols.txt" \
    "${tmp}/${compiler}.undefined"
  while IFS= read -r symbol; do
    readelf -Ws "${object}" |
      awk -v symbol="${symbol}" \
        '$8 == symbol && $5 == "GLOBAL" && $6 == "HIDDEN" { found = 1 }
         END { exit found ? 0 : 1 }'
  done <"${PROJECT_ROOT}/native/tr3_c_restore_bindings_defined_symbols.txt"

  "${compiler}" "${cflags[@]}" "${feature[@]}" "${test_feature[@]}" \
    "${PROJECT_ROOT}/tests/tr3_restore_bindings/test_native_bindings.c" \
    "${PROJECT_ROOT}/native/tr3_c_restore_bindings.c" \
    "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
    -o "${tmp}/test-${compiler}"
  for repetition in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 180s \
      "${tmp}/test-${compiler}" \
      >"${tmp}/${compiler}-${repetition}.stdout" \
      2>"${tmp}/${compiler}-${repetition}.stderr"
    test ! -s "${tmp}/${compiler}-${repetition}.stderr"
  done
  cmp "${tmp}/${compiler}-1.stdout" "${tmp}/${compiler}-2.stdout"
  grep -E '^TR3-C restore bindings PASS: [0-9]+ checks$' \
    "${tmp}/${compiler}-1.stdout" >/dev/null
done
cmp "${tmp}/clang-1.stdout" "${tmp}/gcc-1.stdout"

clang "${cflags[@]}" "${feature[@]}" "${test_feature[@]}" \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/tr3_restore_bindings/test_native_bindings.c" \
  "${PROJECT_ROOT}/native/tr3_c_restore_bindings.c" \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm -o "${tmp}/test-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 300s \
    "${tmp}/test-sanitized" >"${tmp}/sanitized.stdout" \
    2>"${tmp}/sanitized.stderr"
test ! -s "${tmp}/sanitized.stderr"
cmp "${tmp}/clang-1.stdout" "${tmp}/sanitized.stdout"

printf 'TR3-C restore binding gate PASS\n'
