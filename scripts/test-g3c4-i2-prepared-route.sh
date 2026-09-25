#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar python3 timeout; do require_command "${command}"; done
python3 "${PROJECT_ROOT}/scripts/check-g3c4-i2-prepared-route.py"

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-i2-prepared-route-test}"
mkdir -p "${evidence}"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-i2-route.XXXXXX")"
cleanup() {
  local result=$?
  if (( result != 0 )); then
    cat "${temporary_dir}/compile.log" >&2 || true
    cat "${temporary_dir}/runtime.stdout" >&2 || true
  fi
  rm -rf -- "${temporary_dir}"
  return "${result}"
}
trap cleanup EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC
  -fvisibility=hidden -fno-common -fstack-protector-all
  -ffp-contract=off -fexcess-precision=standard -fno-fast-math
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
  -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/src/eshkol_transformer"
)
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/g3c4/i2_prepared_route_native.c" \
  -o "${temporary_dir}/g3c4_i2_native.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${temporary_dir}/p1_identity.o"
for source in data_io checkpoint_io i64_tensor t1_i64_shell \
              n3k_primitives_provider n2_primitives_provider \
              a2_attention_provider; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${temporary_dir}/${source}.o"
done
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/m3t/kernel_fail_allocator.c" \
  -o "${temporary_dir}/kernel_abi.o"
ar rcsD "${temporary_dir}/libg3c4_i2_prepared_route.a" \
  "${temporary_dir}"/*.o

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${temporary_dir}/cache" ESHKOL_CXX_COMPILER="${cxx}" \
  ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 420s "${runner}" \
    --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" \
    -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/native" \
    -L "${temporary_dir}" --lib g3c4_i2_prepared_route \
    "${PROJECT_ROOT}/tests/g3c4/i2_prepared_route_test.esk" \
    -o "${temporary_dir}/route" >"${temporary_dir}/compile.log" 2>&1

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  90s "${temporary_dir}/route" >"${temporary_dir}/runtime.stdout"
grep -E '^G3-C4 I2 prepared route PASS: checks=[1-9][0-9]*$' \
  "${temporary_dir}/runtime.stdout" >/dev/null
cp "${temporary_dir}/compile.log" "${evidence}/compile.log"
cp "${temporary_dir}/runtime.stdout" "${evidence}/runtime.stdout"
cat "${evidence}/runtime.stdout"
printf 'G3-C4 I2 prepared route evidence: %s\n' "${evidence}"
