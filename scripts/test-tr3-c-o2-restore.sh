#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in awk clang cmp comm gcc git grep nm readelf tar timeout; do
  require_command "${command}"
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tr3-c-o2-restore.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    local log
    shopt -s nullglob
    for log in "${tmp}"/*.stdout "${tmp}"/*.stderr; do
      if [[ -s "${log}" ]]; then
        printf 'TR3-C O2 diagnostic %s:\n' "${log}" >&2
        sed -n '1,240p' "${log}" >&2
      fi
    done
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

baseline_ref=1c3745b10d349adcdcb4ae6d845705de76795d20
baseline_root="${tmp}/pre-o2-restore"
mkdir -p "${baseline_root}"
git -C "${PROJECT_ROOT}" cat-file -e "${baseline_ref}^{commit}"
git -C "${PROJECT_ROOT}" archive "${baseline_ref}" \
  native/o2_optimizer.c native/o2_optimizer_internal.h |
  tar -x -C "${baseline_root}"
if cmp -s "${PROJECT_ROOT}/native/o2_optimizer.c" \
    "${baseline_root}/native/o2_optimizer.c"; then
  die "pre-restore O2 baseline unexpectedly matches the current source"
fi

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
  ordinary="${tmp}/${compiler}-ordinary.o"
  baseline="${tmp}/${compiler}-pre-restore.o"
  declaration="${tmp}/${compiler}-declaration.o"
  feature="${tmp}/${compiler}-feature.o"
  step="${tmp}/${compiler}-step.o"
  combined="${tmp}/${compiler}-combined.o"

  if "${compiler}" "${cflags[@]}" -DET_TR3_C_O2_RESTORE_NATIVE \
      -c "${PROJECT_ROOT}/native/o2_optimizer.c" \
      -o "${tmp}/${compiler}-invalid-ungated.o" \
      >"${tmp}/${compiler}-invalid-ungated.stdout" \
      2>"${tmp}/${compiler}-invalid-ungated.stderr"; then
    die "${compiler} accepted O2 restore without the shared matcher gate"
  fi
  grep -F 'requires the shared owned-clone authorizer' \
    "${tmp}/${compiler}-invalid-ungated.stderr" >/dev/null

  (
    cd "${PROJECT_ROOT}"
    "${compiler}" "${cflags[@]}" -c native/o2_optimizer.c -o "${ordinary}"
  )
  (
    cd "${baseline_root}"
    "${compiler}" "${cflags[@]}" -c native/o2_optimizer.c -o "${baseline}"
  )
  cmp "${ordinary}" "${baseline}"
  "${compiler}" "${cflags[@]}" -DET_I2_PRIVATE_OWNED_CLONE_MATCH -c \
    "${PROJECT_ROOT}/native/o2_optimizer.c" -o "${declaration}"
  cmp "${ordinary}" "${declaration}"
  "${compiler}" "${cflags[@]}" -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
    -DET_TR3_C_O2_RESTORE_NATIVE -c \
    "${PROJECT_ROOT}/native/o2_optimizer.c" -o "${feature}"
  "${compiler}" "${cflags[@]}" -DET_TR3_O2_STEP_CLEAR_NATIVE -c \
    "${PROJECT_ROOT}/native/o2_optimizer.c" -o "${step}"
  "${compiler}" "${cflags[@]}" -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
    -DET_TR3_C_O2_RESTORE_NATIVE -DET_TR3_O2_STEP_CLEAR_NATIVE -c \
    "${PROJECT_ROOT}/native/o2_optimizer.c" -o "${combined}"

  for kind in ordinary feature step combined; do
    object="${!kind}"
    symbols defined "${object}" "${tmp}/${compiler}-${kind}.defined"
    symbols undefined "${object}" "${tmp}/${compiler}-${kind}.undefined"
  done
  symbols defined "${baseline}" "${tmp}/${compiler}-baseline.defined"
  symbols undefined "${baseline}" "${tmp}/${compiler}-baseline.undefined"
  cmp "${tmp}/${compiler}-ordinary.defined" \
    "${tmp}/${compiler}-baseline.defined"
  cmp "${tmp}/${compiler}-ordinary.undefined" \
    "${tmp}/${compiler}-baseline.undefined"
  cmp "${PROJECT_ROOT}/native/o2_optimizer_tr3_c_restore_defined_symbols.txt" \
    "${tmp}/${compiler}-feature.defined"
  cmp \
    "${PROJECT_ROOT}/native/o2_optimizer_tr3_c_restore_undefined_symbols.txt" \
    "${tmp}/${compiler}-feature.undefined"
  if grep -E '^et_o2_tr3_c_restore_' \
      "${tmp}/${compiler}-ordinary.defined" >/dev/null; then
    die "ordinary ${compiler} O2 object exposed a restore symbol"
  fi
  test ! -s <(comm -23 "${tmp}/${compiler}-ordinary.defined" \
                       "${tmp}/${compiler}-feature.defined")
  test ! -s <(comm -23 "${tmp}/${compiler}-step.defined" \
                       "${tmp}/${compiler}-combined.defined")
  comm -13 "${tmp}/${compiler}-ordinary.defined" \
    "${tmp}/${compiler}-feature.defined" \
    >"${tmp}/${compiler}-feature-defined-delta"
  comm -13 "${tmp}/${compiler}-step.defined" \
    "${tmp}/${compiler}-combined.defined" \
    >"${tmp}/${compiler}-combined-defined-delta"
  cmp "${tmp}/${compiler}-feature-defined-delta" \
    "${tmp}/${compiler}-combined-defined-delta"
  [[ "$(wc -l <"${tmp}/${compiler}-feature-defined-delta")" == 6 ]]
  while IFS= read -r symbol; do
    [[ "${symbol}" == et_o2_tr3_c_restore_* ]] ||
      die "unexpected O2 restore defined-symbol delta: ${symbol}"
    check_hidden "${feature}" "${symbol}"
  done <"${tmp}/${compiler}-feature-defined-delta"

  test_flags=(-DET_F32_TENSOR_TESTING -DET_O2_TESTING
              -DET_I2_NATIVE_HELPERS_ONLY
              -DET_I2_PRIVATE_OWNED_CLONE_MATCH
              -DET_TR3_C_I2_RESTORE_PRIVATE
              -DET_TR3_C_O2_RESTORE_NATIVE
              -DET_TR3_O2_STEP_CLEAR_NATIVE)
  "${compiler}" "${cflags[@]}" "${test_flags[@]}" \
    "${PROJECT_ROOT}/tests/o2/test_tr3_c_o2_restore.c" \
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
  grep -E '^TR3-C O2 retained-control slope: [0-9]+ bytes/restore$' \
    "${tmp}/${compiler}-1.stdout" >/dev/null
  grep -E '^TR3-C O2 restore PASS: [0-9]+ checks$' \
    "${tmp}/${compiler}-1.stdout" >/dev/null
done

sanitize_flags=(-DET_F32_TENSOR_TESTING -DET_O2_TESTING
                -DET_I2_NATIVE_HELPERS_ONLY
                -DET_I2_PRIVATE_OWNED_CLONE_MATCH
                -DET_TR3_C_I2_RESTORE_PRIVATE
                -DET_TR3_C_O2_RESTORE_NATIVE
                -DET_TR3_O2_STEP_CLEAR_NATIVE)
clang "${cflags[@]}" "${sanitize_flags[@]}" \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/o2/test_tr3_c_o2_restore.c" \
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

clang "${cflags[@]}" "${sanitize_flags[@]}" -O3 \
  "${PROJECT_ROOT}/tests/o2/test_tr3_c_o2_restore.c" \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm -o "${tmp}/test-long-cycles"
ET_TR3_C_O2_CYCLE_MODE=abort \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
    "${tmp}/test-long-cycles" >"${tmp}/long-cycles.stdout" \
    2>"${tmp}/long-cycles.stderr"
test ! -s "${tmp}/long-cycles.stderr"
grep -Fx 'TR3-C O2 cycle batch PASS: 1024 abort, 0 commit' \
  "${tmp}/long-cycles.stdout" >/dev/null
ET_TR3_C_O2_CYCLE_MODE=commit \
  timeout --foreground --signal=TERM --kill-after=5s 1800s \
    "${tmp}/test-long-cycles" >"${tmp}/commit-cycles.stdout" \
    2>"${tmp}/commit-cycles.stderr"
test ! -s "${tmp}/commit-cycles.stderr"
grep -Fx 'TR3-C O2 cycle batch PASS: 0 abort, 8192 commit' \
  "${tmp}/commit-cycles.stdout" >/dev/null

clang "${cflags[@]}" "${sanitize_flags[@]}" -O2 \
  "${PROJECT_ROOT}/tests/o2/test_tr3o_update_clear_adversarial.c" \
  "${PROJECT_ROOT}/native/o2_optimizer.c" \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${tmp}/update-clear-combined"
timeout --foreground --signal=TERM --kill-after=5s 600s \
  "${tmp}/update-clear-combined" >"${tmp}/update-clear.stdout" \
  2>"${tmp}/update-clear.stderr"
test ! -s "${tmp}/update-clear.stderr"
grep -Fx 'TR3-O update-clear adversarial PASS: 37264 checks' \
  "${tmp}/update-clear.stdout" >/dev/null

if grep -R -E 'et_o2_tr3_c_restore_|ET_TR3_C_O2_RESTORE' \
    "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/lib" \
    "${PROJECT_ROOT}/src" >/dev/null; then
  die "TR3-C O2 restore leaked into a public header or Scheme binding"
fi
if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/tr3_c_o2_restore_internal.h" \
    "${PROJECT_ROOT}/native/tr3_c_o2_restore.inc" >/dev/null; then
  die "TR3-C O2 native path contains a forbidden Python/PyTorch reference"
fi

/usr/bin/bash "${PROJECT_ROOT}/scripts/test-tr3-c-i2-restore.sh" >/dev/null
printf 'TR3-C O2 RESTORE PASS: GCC/Clang, exact manifests, I2/#119 coexistence, failure atomicity, terminals, sanitizers\n'
