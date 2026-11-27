#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar python3 rg sha256sum timeout; do require_command "${command}"; done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${IO5_LIVE_QUANTIZED_COMPILER_TIMEOUT_SECONDS:-480}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout must be a positive integer"

mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
scratch="$(mktemp -d "${build}/.io5-live-quantized.XXXXXX")"
trap 'status=$?; if [[ "${status}" == 0 ]]; then rm -rf -- "${scratch}"; else printf "IO5 live quantized failure evidence: %s\n" "${scratch}" >&2; fi' EXIT
mkdir -p "${scratch}/cache"

clean=(env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH
       -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE
       -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX)
"${clean[@]}" CC="${cc}" CXX="${cxx}" bash scripts/build-io5-quant.sh
"${clean[@]}" bash scripts/build-p1-identity.sh "${scratch}/p1" normal trusted

# Reuse the M3 aggregate's native sources and provider builds. Its trusted
# Eshkol root supplies C1, P1, I2 and M3 in a single registry universe.
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
        -O2 -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
        -fexcess-precision=standard -fno-fast-math -frounding-math
        -I include -I native -I src)
sources=(native/data_io.c native/checkpoint_io.c native/kernel_abi.c
         src/eshkol_transformer/m3_i64_integration.c native/t1_i64_shell.c
         src/eshkol_transformer/m3t_f32_integration.c src/eshkol_transformer/m3_model.c)
objects=("${scratch}/i2_helpers.o")
"${clean[@]}" "${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
  -c native/i2_wave2_package_bridge.c -o "${objects[0]}"
for source in "${sources[@]}"; do
  object="${scratch}/$(basename "${source%.c}").o"
  "${clean[@]}" "${cc}" "${cflags[@]}" -c "${source}" -o "${object}"
  objects+=("${object}")
done
for provider in n2 n3k a2; do
  "${clean[@]}" bash "scripts/build-${provider}.sh" "${scratch}/providers/${provider}" normal
done
while IFS= read -r object; do
  objects+=("${scratch}/providers/${object}")
done < native/m3_package_native_objects.txt
ar rcsD "${scratch}/libio5_quantized_m3_native.a" "${objects[@]}"

binary="${scratch}/io5-quantized-m3"
if ! (
  cd "${scratch}"
  timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${clean[@]}" -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_JIT_CACHE=0 \
    ESHKOL_CXX_COMPILER="${cxx}" XDG_CACHE_HOME="${scratch}/cache" \
    "${runner}" --strict-types --optimize 0 --no-stdlib \
    -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src" \
    -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
    -L "${scratch}/p1/trusted" --lib eshkol_transformer_p1_identity \
    -L "${scratch}" --lib io5_quantized_m3_native \
    "${PROJECT_ROOT}/tests/io5/live_quantized_m3.esk" -o "${binary}" \
    >compile.log 2>&1
) || [[ ! -x "${binary}" ]] || rg -q 'ERROR:' "${scratch}/compile.log"; then
  tail -100 "${scratch}/compile.log" >&2
  die "IO5 live quantized M3 AOT compilation failed"
fi
(cd "${scratch}" && sha256sum io5-quantized-m3 libio5_quantized_m3_native.a)
ESHKOL_IO5_QUANT_LIB="${PROJECT_ROOT}/build/io5_quant/libeshkol_transformer_io5_quant.so" \
  ESHKOL_ARENA_POISON=1 IO5_LIVE_QUANTIZED_BRIDGE="${binary}" \
  python3 tests/io5/test_live_quantized_m3.py -v
