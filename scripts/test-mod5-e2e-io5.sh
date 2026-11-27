#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${PROJECT_ROOT}"
for command in ar cmp python3 readelf sha256sum timeout; do
  require_command "${command}"
done
verify_toolchain
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
cc="$(tsv_value "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" cc_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${MOD5_E2E_COMPILER_TIMEOUT_SECONDS:-480}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "compiler timeout must be a positive integer"

mkdir -p "$(project_build_dir)"
build="$(realpath -- "$(project_build_dir)")"
export BUILD_DIR="${build}"
scratch="$(mktemp -d "${build}/.mod5-e2e.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

bash scripts/build-mod5.sh
bash scripts/build-c1.sh
CC="${cc}" bash scripts/build-io5-quant.sh
bash scripts/build-p1-identity.sh "${scratch}/p1" normal trusted

# Use the same trusted I2/C1 roots as the live IO5 witness, plus the MOD5
# provider and its existing AOT bridge. Keep one K1 runtime in the executable.
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wsign-conversion
        -Wshadow -fPIC -fvisibility=hidden -fno-common -fstack-protector-all
        -ffp-contract=off -fexcess-precision=standard -frounding-math
        -I include -I native)
objects=()
for source in i2_wave2_package_bridge data_io kernel_abi i64_tensor t1_i64_shell f32_tensor mod5_eshkol_bridge; do
  object="${scratch}/${source}.o"
  "${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -c "native/${source}.c" -o "${object}"
  objects+=("${object}")
done
ar rcsD "${scratch}/libmod5_e2e_native.a" "${objects[@]}"

for run in a b; do
  output="${scratch}/${run}"
  mkdir -p "${output}/cache"
  if ! (
    cd "${output}"
    timeout --foreground --kill-after=5s "${compiler_timeout}s" \
      env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${output}/cache" \
      "${runner}" --strict-types --optimize 0 --no-stdlib \
        -I "${PROJECT_ROOT}/internal/c1/lib" -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" -I "${PROJECT_ROOT}/src" \
        -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
        -L "${scratch}/p1/trusted" --lib eshkol_transformer_p1_identity \
        -L "${build}/c1" --lib eshkol_transformer_checkpoint_io \
        -L "${scratch}" --lib mod5_e2e_native \
        -L "${PROJECT_ROOT}/build/mod5" --lib eshkol_transformer_mod5 \
        "${PROJECT_ROOT}/tests/mod5/mod5_e2e_c1_io5.esk" -o mod5-e2e >compile.log 2>&1
  ) || [[ ! -x "${output}/mod5-e2e" ]] || grep -Fq 'ERROR:' "${output}/compile.log"; then
    tail -100 "${output}/compile.log" >&2
    die "MOD5 E2E AOT compilation ${run} failed"
  fi
done

readelf -h "${scratch}/a/mod5-e2e" >/dev/null
cmp "${scratch}/a/mod5-e2e" "${scratch}/b/mod5-e2e"
(cd "${scratch}/a" && sha256sum mod5-e2e)
printf 'MOD5 E2E two fresh AOT builds with byte-identical ELF: PASS\n'
MOD5_E2E_AOT="${scratch}/a/mod5-e2e" \
  ESHKOL_IO5_QUANT_LIB="${PROJECT_ROOT}/build/io5_quant/libeshkol_transformer_io5_quant.so" \
  python3 tests/mod5/test_mod5_e2e_io5.py -v
printf 'MOD5 E2E training, C1, lossless, narrow safetensors and mixed Q8_0 routes with fresh model scoring: PASS\n'
