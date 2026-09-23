#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in cmp grep nm python3 timeout; do
  require_command "${command}"
done

resolve_executable_into cc "${CC:-clang}" "TR3-O C compiler"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-tr3-o.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -ffp-contract=off -fexcess-precision=standard -frounding-math
  -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -DET_O2_TESTING -DET_F32_TENSOR_TESTING
)
runtime_sources=(
  "${PROJECT_ROOT}/native/f32_tensor.c"
  "${PROJECT_ROOT}/native/o2_optimizer.c"
  "${PROJECT_ROOT}/native/kernel_abi.c"
)

run_once() {
  local binary="$1" label="$2" expected="$3"
  timeout --foreground --signal=TERM --kill-after=5s 600s \
    "${binary}" >"${temporary_dir}/${label}.stdout"
  grep -Fx "${expected}" "${temporary_dir}/${label}.stdout" >/dev/null
}

run_stateful_twice() {
  local binary="$1" label="$2" expected="$3"
  local run
  for run in 1 2; do
    timeout --foreground --signal=TERM --kill-after=5s 600s \
      "${binary}" >"${temporary_dir}/${label}-${run}.stdout"
  done
  cmp "${temporary_dir}/${label}-1.stdout" \
    "${temporary_dir}/${label}-2.stdout"
  grep -Fx 'TR3-O retained-control slope: outer=96 i2=1112 bytes/update' \
    "${temporary_dir}/${label}-1.stdout" >/dev/null
  grep -Fx 'TR3-O repeated-state digest: e40b45e3746140d1' \
    "${temporary_dir}/${label}-1.stdout" >/dev/null
  grep -Fx "${expected}" "${temporary_dir}/${label}-1.stdout" >/dev/null
}

"${cc}" "${cflags[@]}" -O2 -DET_TR3_O2_STEP_CLEAR_NATIVE \
  "${PROJECT_ROOT}/tests/o2/test_tr3o_update_clear_adversarial.c" \
  "${runtime_sources[@]}" -lm -o "${temporary_dir}/update-clear"
run_stateful_twice "${temporary_dir}/update-clear" update-clear \
  'TR3-O update-clear adversarial PASS: 37264 checks'

"${cc}" "${cflags[@]}" -O1 -g -DET_TR3_O2_STEP_CLEAR_NATIVE \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/o2/test_tr3o_update_clear_adversarial.c" \
  "${runtime_sources[@]}" -lm -o "${temporary_dir}/update-clear-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${temporary_dir}/update-clear-sanitized" \
    >"${temporary_dir}/update-clear-sanitized.stdout"
grep -Fx 'TR3-O retained-control slope: outer=96 i2=1112 bytes/update' \
  "${temporary_dir}/update-clear-sanitized.stdout" >/dev/null
grep -Fx 'TR3-O repeated-state digest: e40b45e3746140d1' \
  "${temporary_dir}/update-clear-sanitized.stdout" >/dev/null
grep -Fx 'TR3-O update-clear adversarial PASS: 37264 checks' \
  "${temporary_dir}/update-clear-sanitized.stdout" >/dev/null

"${cc}" "${cflags[@]}" -O2 \
  "${PROJECT_ROOT}/tests/o2/test_tr3o_i2_plan_coexistence.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -lm \
  -o "${temporary_dir}/i2-plan-coexistence"
run_once "${temporary_dir}/i2-plan-coexistence" i2-plan-coexistence \
  'TR3-O I2 plan coexistence PASS: 99 checks'

for source in test_native_optimizer test_optimizer_native; do
  "${cc}" "${cflags[@]}" -O2 -DET_TR3_O2_STEP_CLEAR_NATIVE \
    "${PROJECT_ROOT}/tests/o2/${source}.c" "${runtime_sources[@]}" -lm \
    -o "${temporary_dir}/${source}"
done
run_once "${temporary_dir}/test_native_optimizer" predecessor-native \
  'O2 native optimizer PASS'
run_once "${temporary_dir}/test_optimizer_native" predecessor-adversarial \
  'O2 native adversarial PASS: 6201 checks'

"${cc}" "${cflags[@]}" -O2 -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_TR3_O2_STEP_CLEAR_BRIDGE \
  "${PROJECT_ROOT}/tests/o2/test_tr3_o2_step_clear_bridge.c" \
  "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
  "${runtime_sources[@]}" -lm \
  -o "${temporary_dir}/step-clear-bridge"
timeout --foreground --signal=TERM --kill-after=5s 60s \
  "${temporary_dir}/step-clear-bridge"

CC="${cc}" PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.o2.test_tr3_o2_step_clear_package

if grep -Ein 'python|pytorch|torch' \
    "${PROJECT_ROOT}/native/tr3_o2_step_clear_internal.h" \
    "${PROJECT_ROOT}/native/tr3_o2_step_clear.inc" \
    "${PROJECT_ROOT}/native/tr3_o2_step_clear_extension.esk"; then
  die "TR3-O native/source runtime path contains a forbidden Python/PyTorch reference"
fi

printf 'TR3-O PASS: update+clear transaction, I2 coexistence, O2 regression, bridge/package-boundary privacy, and sanitizers\n'
