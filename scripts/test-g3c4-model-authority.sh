#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar python3 timeout; do require_command "${command}"; done
python3 "${PROJECT_ROOT}/scripts/check-g3c4-model-authority.py"

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
evidence="${1:-$(project_build_dir)/g3c4-model-authority-test}"
mkdir -p "${evidence}"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-g3c4-model.XXXXXX")"
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
  -I "$(eshkol_source_dir)/inc"
)
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_G3C4_I2_CONSTRUCTION_PRIVATE -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${temporary_dir}/i2_native_helpers.o"
"${cc}" "${cflags[@]}" -DET_G3C4_NATIVE_OWNER_PRIVATE \
  -c "${PROJECT_ROOT}/src/eshkol_transformer/m3t_f32_integration.c" \
  -o "${temporary_dir}/m3t_f32_integration.o"
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/g3c4/model_authority_native.c" \
  -o "${temporary_dir}/g3c4_model_owner.o"
for source in data_io checkpoint_io kernel_abi t1_i64_shell \
              n3k_primitives_provider n2_primitives_provider \
              a2_attention_provider; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${temporary_dir}/${source}.o"
done
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${temporary_dir}/p1_identity.o"
for source in m3_i64_integration m3_model; do
  "${cc}" "${cflags[@]}" \
    -c "${PROJECT_ROOT}/src/eshkol_transformer/${source}.c" \
    -o "${temporary_dir}/${source}.o"
done
ar rcsD "${temporary_dir}/libg3c4_model_authority.a" \
  "${temporary_dir}"/*.o

env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${temporary_dir}/cache" ESHKOL_CXX_COMPILER="${cxx}" \
  ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  timeout --foreground --signal=TERM --kill-after=5s 600s "${runner}" \
    --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" \
    -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/native" \
    -L "${temporary_dir}" --lib g3c4_model_authority \
    "${PROJECT_ROOT}/tests/g3c4/model_authority_test.esk" \
    -o "${temporary_dir}/model-authority" \
    >"${temporary_dir}/compile.log" 2>&1

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
  120s "${temporary_dir}/model-authority" \
  >"${temporary_dir}/runtime.stdout"
grep -E '^G3-C4 model authority PASS: checks=[1-9][0-9]*$' \
  "${temporary_dir}/runtime.stdout" >/dev/null
cp "${temporary_dir}/compile.log" "${evidence}/compile.log"
cp "${temporary_dir}/runtime.stdout" "${evidence}/runtime.stdout"
cat "${evidence}/runtime.stdout"
printf 'G3-C4 model authority evidence: %s\n' "${evidence}"
