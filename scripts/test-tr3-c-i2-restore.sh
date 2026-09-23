#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in awk clang cmp comm gcc grep nm readelf timeout; do
  require_command "${command}"
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tr3-c-i2-restore.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    local log
    shopt -s nullglob
    for log in "${tmp}"/*.stdout "${tmp}"/*.stderr; do
      if [[ -s "${log}" ]]; then
        printf 'TR3-C I2 diagnostic %s:\n' "${log}" >&2
        sed -n '1,240p' "${log}" >&2
      fi
    done
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -O0
        -ffp-contract=off -fexcess-precision=standard -frounding-math
        -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
        -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")

symbols() {
  local mode=$1 object=$2 output=$3
  nm -g "--${mode}-only" --format=posix "${object}" |
    awk '{ print $1 }' |
    grep -v '^_GLOBAL_OFFSET_TABLE_$' |
    LC_ALL=C sort >"${output}"
}

check_hidden() {
  local object=$1 symbol=$2
  readelf -Ws "${object}" |
    awk -v symbol="${symbol}" \
      '$8 == symbol && $5 == "GLOBAL" && $6 == "HIDDEN" { found = 1 }
       END { exit found ? 0 : 1 }'
}

for compiler in clang gcc; do
  ordinary_f32="${tmp}/${compiler}-f32-ordinary.o"
  ordinary_f32_parity="${tmp}/${compiler}-f32-ordinary-parity.o"
  feature_f32="${tmp}/${compiler}-f32-feature.o"
  ordinary_bridge="${tmp}/${compiler}-bridge-ordinary.o"
  declaration_bridge="${tmp}/${compiler}-bridge-declaration-only.o"
  feature_bridge="${tmp}/${compiler}-bridge-feature.o"

  if "${compiler}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
      -DET_TR3_C_I2_RESTORE_PRIVATE \
      -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
      -o "${tmp}/${compiler}-invalid-ungated.o" \
      >"${tmp}/${compiler}-invalid-ungated.stdout" \
      2>"${tmp}/${compiler}-invalid-ungated.stderr"; then
    die "${compiler} accepted restore without the shared matcher gate"
  fi
  grep -F 'requires the shared owned-clone authorizer' \
    "${tmp}/${compiler}-invalid-ungated.stderr" >/dev/null

  "${compiler}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/f32_tensor.c" \
    -o "${ordinary_f32}"
  "${compiler}" "${cflags[@]}" -UET_F32_TENSOR_TESTING \
    -c "${PROJECT_ROOT}/native/f32_tensor.c" -o "${ordinary_f32_parity}"
  cmp "${ordinary_f32}" "${ordinary_f32_parity}"
  "${compiler}" "${cflags[@]}" -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
    -c "${PROJECT_ROOT}/native/f32_tensor.c" -o "${feature_f32}"
  "${compiler}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    -o "${ordinary_bridge}"
  "${compiler}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
    -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    -o "${declaration_bridge}"
  cmp "${ordinary_bridge}" "${declaration_bridge}"
  "${compiler}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_I2_PRIVATE_OWNED_CLONE_MATCH -DET_TR3_C_I2_RESTORE_PRIVATE \
    -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    -o "${feature_bridge}"

  for kind in ordinary_f32 feature_f32 ordinary_bridge feature_bridge; do
    object="${!kind}"
    symbols defined "${object}" "${tmp}/${compiler}-${kind}.defined"
    symbols undefined "${object}" "${tmp}/${compiler}-${kind}.undefined"
  done

  cmp "${PROJECT_ROOT}/native/f32_tensor_defined_symbols.txt" \
    "${tmp}/${compiler}-ordinary_f32.defined"
  cmp "${PROJECT_ROOT}/native/f32_tensor_undefined_symbols.txt" \
    "${tmp}/${compiler}-ordinary_f32.undefined"
  cmp "${PROJECT_ROOT}/native/f32_tensor_owned_match_defined_symbols.txt" \
    "${tmp}/${compiler}-feature_f32.defined"
  cmp "${PROJECT_ROOT}/native/f32_tensor_owned_match_undefined_symbols.txt" \
    "${tmp}/${compiler}-feature_f32.undefined"
  cmp "${PROJECT_ROOT}/native/tr3_c_i2_restore_bridge_defined_symbols.txt" \
    "${tmp}/${compiler}-feature_bridge.defined"
  cmp "${PROJECT_ROOT}/native/tr3_c_i2_restore_bridge_undefined_symbols.txt" \
    "${tmp}/${compiler}-feature_bridge.undefined"

  if grep -E '^et_tr3_c_i2_|^et_i2_private_owned_clone_match_v1$' \
      "${tmp}/${compiler}-ordinary_f32.defined" \
      "${tmp}/${compiler}-ordinary_bridge.defined" >/dev/null; then
    die "ordinary ${compiler} I2 objects expose a restore symbol"
  fi
  [[ "$(comm -13 "${tmp}/${compiler}-ordinary_f32.defined" \
                    "${tmp}/${compiler}-feature_f32.defined")" == \
      et_i2_private_owned_clone_match_v1 ]] ||
    die "${compiler} matcher gate changed an unexpected defined symbol"
  test ! -s <(comm -23 "${tmp}/${compiler}-ordinary_f32.defined" \
                       "${tmp}/${compiler}-feature_f32.defined")
  test ! -s <(comm -3 "${tmp}/${compiler}-ordinary_f32.undefined" \
                      "${tmp}/${compiler}-feature_f32.undefined")
  grep -Fx et_i2_private_owned_clone_match_v1 \
    "${tmp}/${compiler}-feature_bridge.undefined" >/dev/null

  check_hidden "${feature_f32}" et_i2_private_owned_clone_match_v1
  while IFS= read -r symbol; do
    [[ "${symbol}" == et_tr3_c_i2_* ]] || continue
    check_hidden "${feature_bridge}" "${symbol}"
  done <"${tmp}/${compiler}-feature_bridge.defined"

  "${compiler}" "${cflags[@]}" -DET_F32_TENSOR_TESTING \
    -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
    "${PROJECT_ROOT}/tests/i2/test_tr3_c_i2_restore.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
    -o "${tmp}/test-${compiler}"
  for repetition in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 90s \
      "${tmp}/test-${compiler}" \
      >"${tmp}/${compiler}-${repetition}.stdout" \
      2>"${tmp}/${compiler}-${repetition}.stderr"
    test ! -s "${tmp}/${compiler}-${repetition}.stderr"
  done
  cmp "${tmp}/${compiler}-1.stdout" "${tmp}/${compiler}-2.stdout"
  grep -Fx 'TR3-C I2 retained-control slope: 64 bytes/restore' \
    "${tmp}/${compiler}-1.stdout" >/dev/null
  grep -E '^TR3-C I2 restore PASS: [0-9]+ checks$' \
    "${tmp}/${compiler}-1.stdout" >/dev/null

  "${compiler}" "${cflags[@]}" -O2 -fstrict-aliasing \
    -DET_F32_TENSOR_TESTING -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
    "${PROJECT_ROOT}/tests/i2/test_tr3_c_i2_restore.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
    -o "${tmp}/test-optimized-${compiler}"
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${tmp}/test-optimized-${compiler}" \
    >"${tmp}/${compiler}-optimized.stdout" \
    2>"${tmp}/${compiler}-optimized.stderr"
  test ! -s "${tmp}/${compiler}-optimized.stderr"
  cmp "${tmp}/${compiler}-1.stdout" "${tmp}/${compiler}-optimized.stdout"

  "${compiler}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_F32_TENSOR_TESTING -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
    -DET_TR3_C_I2_RESTORE_PRIVATE \
    "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/tests/i2/test_tr3_c_checked_bridge.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" \
    "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
    -o "${tmp}/test-checked-${compiler}"
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${tmp}/test-checked-${compiler}" \
    >"${tmp}/${compiler}-checked.stdout" \
    2>"${tmp}/${compiler}-checked.stderr"
  test ! -s "${tmp}/${compiler}-checked.stderr"
  grep -E '^TR3-C checked I2 bridge PASS: [0-9]+ checks$' \
    "${tmp}/${compiler}-checked.stdout" >/dev/null
done

"clang" "${cflags[@]}" -DET_F32_TENSOR_TESTING \
  -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/i2/test_tr3_c_i2_restore.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${tmp}/test-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${tmp}/test-sanitized" >"${tmp}/sanitized.stdout" \
    2>"${tmp}/sanitized.stderr"
test ! -s "${tmp}/sanitized.stderr"
cmp "${tmp}/clang-1.stdout" "${tmp}/sanitized.stdout"

if grep -R -E 'et_tr3_c_i2_|et_i2_private_owned_clone_match_v1' \
    "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/lib" \
    "${PROJECT_ROOT}/src" >/dev/null; then
  die "TR3-C I2 restore leaked into a public header or Scheme binding"
fi
if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/tr3_c_i2_restore_internal.h" \
    "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    "${PROJECT_ROOT}/native/f32_tensor.c" >/dev/null; then
  die "TR3-C I2 native path contains a forbidden Python/PyTorch reference"
fi

printf 'TR3-C I2 RESTORE PASS: GCC/Clang, exact manifests, isolation, negatives, failure atomicity, terminals, sanitizers\n'
