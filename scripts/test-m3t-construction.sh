#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar timeout; do require_command "${command}"; done
cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-m3t-construction.XXXXXX")"
cleanup() {
  local result=$?
  if (( result != 0 )); then
    cat "${tmp}/compile.log" >&2 || true
    if [[ -f "${tmp}/runtime.stdout" ]]; then cat "${tmp}/runtime.stdout" >&2; fi
  fi
  rm -rf -- "${tmp}"
  return "${result}"
}
trap cleanup EXIT
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC -fvisibility=hidden
        -fno-common -ffp-contract=off -fexcess-precision=standard -frounding-math
        -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
        -I "${PROJECT_ROOT}/src" -DET_F32_TENSOR_TESTING)
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" -o "${tmp}/i2_bridge.o"
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell \
              n3k_primitives_provider n2_primitives_provider a2_attention_provider; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${tmp}/${source}.o"
done
for source in m3t_transport m3t_f32_integration; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/src/eshkol_transformer/${source}.c" \
    -o "${tmp}/${source}.o"
done
"${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/tests/p1/i2_construction_test_bridge.c" \
  -o "${tmp}/test_bridge.o"
ar rcsD "${tmp}/libm3t_construction_test.a" "${tmp}"/*.o
(cd "${tmp}" && env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
  ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${tmp}/cache" \
  ESHKOL_CXX_COMPILER="${cxx}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 420s "${runner}" \
    --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/p1/lib" -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -L "${tmp}" --lib m3t_construction_test \
    "${PROJECT_ROOT}/tests/p1/i2_construction_test.esk" -o "${tmp}/construction" \
    >"${tmp}/compile.log" 2>&1)
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  60s "${tmp}/construction" >"${tmp}/runtime.stdout"
grep -E '^I2 construction PASS: [0-9]+$' "${tmp}/runtime.stdout" >/dev/null
cat "${tmp}/runtime.stdout"
