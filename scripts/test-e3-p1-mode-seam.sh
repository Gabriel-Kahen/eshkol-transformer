#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp env grep python3 sha256sum timeout; do
  require_command "${command}"
done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="${1:-$(project_build_dir)/e3-p1-mode-seam}"
[[ ! -e "${out}" ]] || die "E3 P1 mode-seam output directory must be fresh"
mkdir -p "${out}/source/transformer" "${out}/native"
cd "${PROJECT_ROOT}"
sha256sum -c "${PROJECT_ROOT}/tests/e3_p1/mode_seam_sources.sha256" \
  >"${out}/source-pins.stdout"
python3 "${PROJECT_ROOT}/scripts/generate-e3-p1-mode-test-variant.py" \
  --output "${out}/source/transformer/module.esk" \
  >"${out}/variant-sha256.txt"
python3 "${PROJECT_ROOT}/scripts/generate-e3-p1-mode-test-variant.py" \
  --output "${out}/source/transformer/module.esk" --check \
  >"${out}/variant-check-sha256.txt"
cmp "${out}/variant-sha256.txt" "${out}/variant-check-sha256.txt"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
        -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
        -fexcess-precision=standard -frounding-math
        -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
        -I "${PROJECT_ROOT}/src")
compile_native() {
  local source=$1 object=$2
  shift 2
  "${cc}" "${cflags[@]}" "$@" -c "${PROJECT_ROOT}/${source}" \
    -o "${out}/native/${object}" \
    >"${out}/native/${object}.stdout" \
    2>"${out}/native/${object}.stderr"
}
compile_native native/a2_attention_provider.c a2_attention_provider.o
compile_native native/checkpoint_io.c checkpoint_io.o
compile_native native/data_io.c data_io.o
compile_native native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD
compile_native native/kernel_abi.c kernel_abi.o
compile_native src/eshkol_transformer/m3_i64_integration.c m3_i64_integration.o
compile_native src/eshkol_transformer/m3_model.c m3_model.o
compile_native src/eshkol_transformer/m3t_f32_integration.c m3t_f32_integration.o \
  -DET_F32_TENSOR_TESTING
compile_native native/n2_primitives_provider.c n2_primitives_provider.o
compile_native native/n3k_primitives_provider.c n3k_primitives_provider.o
compile_native native/p1_identity.c p1_identity.o \
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1
compile_native native/t1_i64_shell.c t1_i64_shell.o
ar rcsD "${out}/native/libe3_p1_mode_seam_test.a" "${out}"/native/*.o

compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-1200}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] ||
  die "E3 P1 mode-seam compiler timeout must be positive"
for cache in a b; do
  eshkol_args=(--strict-types --no-stdlib -O 2
    -I "${out}/source" -I "${PROJECT_ROOT}/native"
    -I "${PROJECT_ROOT}/internal/p1/lib"
    -I "${PROJECT_ROOT}/internal/c1/lib"
    -I "${PROJECT_ROOT}/internal/t1/lib"
    -I "${PROJECT_ROOT}/src"
    -L "${out}/native" --lib e3_p1_mode_seam_test)
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
    -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES \
    -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
    -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER="${cxx}" \
    XDG_CACHE_HOME="${out}/cache-object-${cache}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${compiler_timeout}s" "${runner}" "${eshkol_args[@]}" \
      --compile-only --emit-depfile "${out}/deps-${cache}.d" \
      "${PROJECT_ROOT}/tests/e3_p1/mode_seam_runtime.esk" \
      -o "${out}/object-${cache}.o" \
      >"${out}/compile-object-${cache}.stdout" \
      2>"${out}/compile-object-${cache}.stderr"
  grep -F "${out}/source/transformer/module.esk" \
    "${out}/deps-${cache}.d" >/dev/null ||
    die "E3 P1 mode-seam build did not resolve the generated P1 variant"
  if grep -F "${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk" \
      "${out}/deps-${cache}.d" >/dev/null; then
    die "E3 P1 mode-seam build also resolved canonical P1"
  fi
  sed '1s/^[^:]*:/<TARGET>:/' "${out}/deps-${cache}.d" \
    >"${out}/deps-${cache}.normalized.d"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
    -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES \
    -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
    -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER="${cxx}" \
    XDG_CACHE_HOME="${out}/cache-link-${cache}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${compiler_timeout}s" "${runner}" "${eshkol_args[@]}" \
      "${PROJECT_ROOT}/tests/e3_p1/mode_seam_runtime.esk" \
      -o "${out}/runtime-${cache}" \
      >"${out}/link-${cache}.stdout" \
      2>"${out}/link-${cache}.stderr"
done
cmp "${out}/object-a.o" "${out}/object-b.o"
cmp "${out}/runtime-a" "${out}/runtime-b"
cmp "${out}/deps-a.normalized.d" "${out}/deps-b.normalized.d"
for cache in a b; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    120s "${out}/runtime-${cache}" \
    >"${out}/runtime-${cache}.stdout" \
    2>"${out}/runtime-${cache}.stderr"
  [[ ! -s "${out}/runtime-${cache}.stderr" ]] ||
    die "E3 P1 mode-seam runtime stderr is not empty"
  grep -Eq '^E3-P1-SEAM-PASS [1-9][0-9]*$' \
    "${out}/runtime-${cache}.stdout" ||
    die "E3 P1 mode-seam runtime did not pass"
done
cmp "${out}/runtime-a.stdout" "${out}/runtime-b.stdout"
printf 'E3 P1 test-only mode seam passed: %s\n' "${out}"
