#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp nm rg sed sort timeout tr; do
  require_command "${command}"
done

cc="${CC:-/usr/bin/clang}"
cxx="${CXX:-/usr/bin/clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-model-encode.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,240p' {} \; >&2 || true
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common
  -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -DET_F32_TENSOR_TESTING
)

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_C2_I2_MODEL_COPY \
  -MMD -MF "${runtime}/i2_wave2_package_bridge.d" \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${runtime}/i2_wave2_package_bridge.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${runtime}/p1_identity.o"
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell \
              f32_tensor; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/c2/c2_model_encode_test_bridge.c" \
  -o "${runtime}/c2_model_encode_test_bridge.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_model_encode.a" \
  "${runtime}"/*.o

# The opt-in bridge exposes exactly the already accepted C2-private seam, and
# its ordinary I2 build exposes none of it.
nm -g --defined-only --format=posix \
  "${runtime}/i2_wave2_package_bridge.o" | \
  awk '$1 ~ /^et_c2_/ { print $1 }' | LC_ALL=C sort \
  >"${tmp}/c2-native-symbols"
cmp "${PROJECT_ROOT}/native/c2_i2_model_copy_defined_symbols.txt" \
  "${tmp}/c2-native-symbols"
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${tmp}/i2-without-c2.o"
if nm -g --defined-only --format=posix "${tmp}/i2-without-c2.o" | \
    rg '^et_c2_[^[:space:]]+[[:space:]]' >/dev/null; then
  die "ordinary I2 bridge exposed the C2-private model-copy seam"
fi

sed -e 's/^[^:]*://' -e 's/\\//g' \
  "${runtime}/i2_wave2_package_bridge.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${tmp}/native-source-closure"
cmp "${PROJECT_ROOT}/native/c2_i2_model_copy_source_closure.txt" \
  "${tmp}/native-source-closure"

# Fix the Eshkol adapter's complete private definition and source surfaces.
sed -n -E \
  's/^\(define \(?(c2-model-[^ )]+).*/\1/p' \
  "${PROJECT_ROOT}/native/c2_model_encode_extension.esk" | \
  LC_ALL=C sort >"${tmp}/adapter-definitions"
cmp "${PROJECT_ROOT}/tests/c2/expected/c2_model_encode_definitions.txt" \
  "${tmp}/adapter-definitions"
if rg '^\(provide|k2|K2' \
    "${PROJECT_ROOT}/native/c2_model_encode_extension.esk" \
    "${PROJECT_ROOT}/native/c2_model_encode_source_closure.txt" >/dev/null; then
  die "C2 model encoder published a facade or composed forbidden K2 input"
fi
{
  cat "${PROJECT_ROOT}/native/i2_wave2_source_closure.txt"
  printf '%s\n' native/c2_model_encode_extension.esk
} >"${tmp}/expected-source-closure"
cmp "${PROJECT_ROOT}/native/c2_model_encode_source_closure.txt" \
  "${tmp}/expected-source-closure"

compile_probe() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/probe-${label}"
  (cd "${tmp}/probe-${label}" && \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-probe-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
      ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 360s "${runner}" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "${tmp}/probe-${label}/model.d" \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" \
        "${PROJECT_ROOT}/tests/c2/c2_model_encode_compile_probe.esk" \
        -o "${tmp}/probe-${label}/model.o" \
        >"${tmp}/probe-${label}/compile.stdout" \
        2>"${tmp}/probe-${label}/compile.stderr")
  test ! -s "${tmp}/probe-${label}/compile.stderr"
  nm -u --format=posix "${tmp}/probe-${label}/model.o" | \
    awk '{ print $1 }' | rg '^et_c2_' | LC_ALL=C sort -u \
    >"${tmp}/probe-${label}/c2-undefined"
  cmp "${PROJECT_ROOT}/native/c2_i2_model_copy_defined_symbols.txt" \
    "${tmp}/probe-${label}/c2-undefined"
  sed -e 's/^[^:]*://' -e 's/\\//g' \
    "${tmp}/probe-${label}/model.d" | tr -s '[:space:]' '\n' | \
    grep -F "${PROJECT_ROOT}/" | sed "s#^${PROJECT_ROOT}/##" | \
    grep -v '^tests/c2/c2_model_encode_compile_probe.esk$' \
    >"${tmp}/probe-${label}/source-closure"
  cmp "${PROJECT_ROOT}/native/c2_model_encode_source_closure.txt" \
    "${tmp}/probe-${label}/source-closure"
}

compile_runtime() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/${label}"
  (cd "${tmp}/${label}" && \
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
      ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 420s "${runner}" \
        --strict-types --optimize 0 --no-stdlib \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" -L "${runtime}" \
        --lib eshkol_transformer_c2_model_encode \
        "${PROJECT_ROOT}/tests/c2/c2_model_encode_runtime.esk" \
        -o "${tmp}/${label}/model-encode" \
        >"${tmp}/${label}/compile.stdout" \
        2>"${tmp}/${label}/compile.stderr")
  test ! -s "${tmp}/${label}/compile.stderr"
  ESHKOL_ARENA_POISON=1 \
    timeout --foreground --signal=TERM --kill-after=5s 180s \
    "${tmp}/${label}/model-encode" >"${tmp}/${label}/run.stdout" \
    2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  grep -E '^C2 MODEL ENCODE PASS: [0-9]+ checks$' \
    "${tmp}/${label}/run.stdout" >/dev/null
}

compile_probe clang-a "${cxx}"
compile_probe clang-b "${cxx}"
cmp "${tmp}/probe-clang-a/model.o" "${tmp}/probe-clang-b/model.o"
compile_runtime clang-a "${cxx}"
compile_runtime clang-b "${cxx}"
cmp "${tmp}/clang-a/model-encode" "${tmp}/clang-b/model-encode"
cmp "${tmp}/clang-a/run.stdout" "${tmp}/clang-b/run.stdout"

if [[ -x /usr/bin/gcc && -x /usr/bin/g++ ]]; then
  /usr/bin/gcc "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
    -DET_C2_I2_MODEL_COPY \
    -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
    -o "${tmp}/gcc-i2-wave2-package-bridge.o"
  /usr/bin/gcc "${cflags[@]}" \
    -c "${PROJECT_ROOT}/tests/c2/c2_model_encode_test_bridge.c" \
    -o "${tmp}/gcc-test-bridge.o"
  compile_probe gcc /usr/bin/g++
  compile_runtime gcc /usr/bin/g++
  cmp "${tmp}/clang-a/run.stdout" "${tmp}/gcc/run.stdout"
fi

# The native no-borrow copy boundary gets its full hostile/failpoint suite
# under both sanitizers; the Eshkol adapter above is strict AOT exercised.
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_C2_I2_MODEL_COPY -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  "${PROJECT_ROOT}/native/f32_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" \
  "${PROJECT_ROOT}/tests/i2/test_c2_model_copy.c" \
  -o "${tmp}/model-copy-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 180s \
  "${tmp}/model-copy-sanitized" >"${tmp}/sanitized.stdout"
grep -Fx 'C2 I2 model copy PASS: 225 checks' \
  "${tmp}/sanitized.stdout" >/dev/null

printf 'C2 MODEL ENCODE PASS: exact C1 bytes/aliases, malformed and atomic failures, flat I2 counters, exact private surfaces, Clang/GCC/C++, ASan/UBSan\n'
