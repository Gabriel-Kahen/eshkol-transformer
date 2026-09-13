#!/usr/bin/bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp diff nm python3 rg sed sort timeout tr; do
  require_command "${command}"
done

cc="${CC:-/usr/bin/clang}"
cxx="${CXX:-/usr/bin/clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-checkpoint-load.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,280p' {} \; >&2 || true
  fi
  rm -rf -- "${tmp}"
  return "${status}"
}
trap cleanup EXIT

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffp-contract=off
  -fexcess-precision=standard -frounding-math -fPIC -fvisibility=hidden
  -fno-common -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native"
)
test_flags=(
  -DET_C2_CHECKPOINT_LOAD_TESTING -DET_C2_CHECKPOINT_CORE_TESTING
  -DET_C2_CHECKPOINT_READER_TESTING -DET_CHECKPOINT_IO_TESTING
)

for label in a b; do
  PYTHONDONTWRITEBYTECODE=1 python3 \
    "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_load_fixtures.py" \
    "${tmp}/fixtures-${label}"
done
diff -ru "${tmp}/fixtures-a" "${tmp}/fixtures-b"
fixture_args=(
  "${tmp}/fixtures-a/valid.c2"
  "${tmp}/fixtures-a/nonf32-buffer.c2"
  "${tmp}/fixtures-a/nonf32-i64-buffer.c2"
  "${tmp}/fixtures-a/larger-valid.c2"
  "${tmp}/fixtures-a/exact-valid.c2"
)

bridge_sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_load_bridge.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/native/checkpoint_io.c"
)
build_bridge() {
  local compiler=$1 output=$2
  "${compiler}" "${cflags[@]}" "${test_flags[@]}" \
    "${bridge_sources[@]}" \
    "${PROJECT_ROOT}/tests/c2/test_checkpoint_load_bridge.c" -o "${output}"
}
build_bridge "${cc}" "${tmp}/bridge-clang"
"${tmp}/bridge-clang" "${fixture_args[@]}" >"${tmp}/bridge-clang.stdout"
rg -x 'C2 checkpoint load bridge PASS: 1163 checks' \
  "${tmp}/bridge-clang.stdout" >/dev/null

for compiler in "${cxx}" /usr/bin/g++; do
  if [[ -x "${compiler}" ]]; then
    "${compiler}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
      -I "${PROJECT_ROOT}/native" \
      "${PROJECT_ROOT}/tests/c2/checkpoint_load_bridge_header_cpp.cpp" \
      -o "${tmp}/header-$(basename "${compiler}")"
    "${tmp}/header-$(basename "${compiler}")"
    "${compiler}" -x c++ -std=c++17 -Wall -Wextra -Werror -Wpedantic \
      -Wconversion -Wsign-conversion -Wshadow -I "${PROJECT_ROOT}/native" \
      -c "${PROJECT_ROOT}/native/c2_checkpoint_load_bridge.c" \
      -o "${tmp}/bridge-$(basename "${compiler}").o"
  fi
done
if [[ -x /usr/bin/gcc ]]; then
  build_bridge /usr/bin/gcc "${tmp}/bridge-gcc"
  "${tmp}/bridge-gcc" "${fixture_args[@]}" >"${tmp}/bridge-gcc.stdout"
  cmp "${tmp}/bridge-clang.stdout" "${tmp}/bridge-gcc.stdout"
fi

"${cc}" "${cflags[@]}" -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_load_bridge.c" \
  -o "${tmp}/bridge-production.o"
nm -g --defined-only --format=posix "${tmp}/bridge-production.o" | \
  awk '$1 ~ /^et_/ { print $1 }' | LC_ALL=C sort >"${tmp}/symbols"
cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_load_bridge_symbols.txt" \
  "${tmp}/symbols"

"${cc}" -std=c11 -I "${PROJECT_ROOT}/native" -MM \
  "${bridge_sources[@]}" | sed -e 's/^[^:]*://' -e 's/\\//g' | \
  tr -s '[:space:]' '\n' | sed "s#^${PROJECT_ROOT}/##" | \
  rg '^native/' | LC_ALL=C sort -u >"${tmp}/native-closure"
LC_ALL=C sort -u \
  "${PROJECT_ROOT}/native/c2_checkpoint_load_native_source_closure.txt" \
  >"${tmp}/expected-native-closure"
cmp "${tmp}/expected-native-closure" "${tmp}/native-closure"

compile_probe() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/probe-${label}"
  (cd "${tmp}/probe-${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/probe-cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 720s "${runner}" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "${tmp}/probe-${label}/load.d" \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t2/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/internal/d2/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" \
        "${PROJECT_ROOT}/tests/c2/c2_checkpoint_load_compile_probe.esk" \
        -o "${tmp}/probe-${label}/load.o" \
        >"${tmp}/probe-${label}/compile.stdout" \
        2>"${tmp}/probe-${label}/compile.stderr")
  test ! -s "${tmp}/probe-${label}/compile.stderr"
  sed -e 's/^[^:]*://' -e 's/\\//g' "${tmp}/probe-${label}/load.d" | \
    tr -s '[:space:]' '\n' | rg "^${PROJECT_ROOT}/" | \
    sed "s#^${PROJECT_ROOT}/##" | \
    rg -v '^tests/c2/c2_checkpoint_load_compile_probe\.esk$' \
    >"${tmp}/probe-${label}/source-closure"
  cmp "${PROJECT_ROOT}/native/c2_checkpoint_load_source_closure.txt" \
    "${tmp}/probe-${label}/source-closure"
  nm -u --format=posix "${tmp}/probe-${label}/load.o" | \
    awk '{ print $1 }' | rg '^et_c2_private_checkpoint_load_' | \
    LC_ALL=C sort -u >"${tmp}/probe-${label}/load-symbols"
  cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_load_bridge_symbols.txt" \
    "${tmp}/probe-${label}/load-symbols"
  if nm -u --format=posix "${tmp}/probe-${label}/load.o" | \
      rg 'et_c2_private_i2_state_owned_copy_bytes_v1' >/dev/null; then
    die "LOAD closure composed the SAVE-only model copy symbol"
  fi
}
compile_probe clang-a "${cxx}"
compile_probe clang-b "${cxx}"
cmp "${tmp}/probe-clang-a/load.o" "${tmp}/probe-clang-b/load.o"
if [[ -x /usr/bin/g++ ]]; then compile_probe gcc /usr/bin/g++; fi

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
runtime_cflags=("${cflags[@]}" -DET_F32_TENSOR_TESTING -DET_O2_TESTING)
"${cc}" "${runtime_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY -c \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" -o "${runtime}/i2.o"
"${cc}" "${runtime_cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE -c \
  "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" -o "${runtime}/o2.o"
"${cc}" "${runtime_cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -DET_P1_TEST_HOOKS=1 -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${runtime}/p1.o"
for source in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor \
              d2_native o2_optimizer c2_x1_canonical c2_checkpoint_format; do
  "${cc}" "${runtime_cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${runtime_cflags[@]}" -DET_CHECKPOINT_IO_TESTING -c \
  "${PROJECT_ROOT}/native/checkpoint_io.c" -o "${runtime}/checkpoint_io.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_READER_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_reader.c" -o "${runtime}/reader.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_CORE_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_core.c" -o "${runtime}/core.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_LOAD_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_load_bridge.c" -o "${runtime}/load.o"
"${cc}" "${runtime_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_test_bridge.c" \
  -o "${runtime}/test.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_load.a" "${runtime}"/*.o

compile_runtime() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/${label}"
  (cd "${tmp}/${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 900s "${runner}" \
        --strict-types --optimize 0 --no-stdlib \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t2/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/internal/d2/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" -L "${runtime}" \
        --lib eshkol_transformer_c2_load \
        "${PROJECT_ROOT}/tests/c2/c2_checkpoint_load_runtime.esk" \
        -o "${tmp}/${label}/load" >"${tmp}/${label}/compile.stdout" \
        2>"${tmp}/${label}/compile.stderr")
  test ! -s "${tmp}/${label}/compile.stderr"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    360s "${tmp}/${label}/load" \
      "${tmp}/fixtures-a/valid.c2" "${tmp}/fixtures-a/exact-valid.c2" \
      "${tmp}/fixtures-a/nonf32-buffer.c2" \
      "${tmp}/fixtures-a/nonf32-i64-buffer.c2" \
      "${PROJECT_ROOT}/tests/x1/fixtures/resolved_minimal_v1.json" \
      >"${tmp}/${label}/run.stdout" 2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  rg -x 'C2 PRIVATE CHECKPOINT LOAD PASS: 234 checks' \
    "${tmp}/${label}/run.stdout" >/dev/null
}
compile_runtime clang-a "${cxx}"
compile_runtime clang-b "${cxx}"
cmp "${tmp}/clang-a/load" "${tmp}/clang-b/load"
cmp "${tmp}/clang-a/run.stdout" "${tmp}/clang-b/run.stdout"
if [[ -x /usr/bin/g++ ]]; then
  compile_runtime gcc /usr/bin/g++
  cmp "${tmp}/clang-a/run.stdout" "${tmp}/gcc/run.stdout"
fi

"${cc}" "${cflags[@]}" "${test_flags[@]}" -O1 \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${bridge_sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/test_checkpoint_load_bridge.c" \
  -o "${tmp}/bridge-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${tmp}/bridge-san" "${fixture_args[@]}" >"${tmp}/bridge-san.stdout"
cmp "${tmp}/bridge-clang.stdout" "${tmp}/bridge-san.stdout"

for gate in test-c2-format.sh test-c2-core.sh test-c2-checkpoint-inspect.sh \
            test-c2-o2-encode.sh test-c2-training-state-owner.sh; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/${gate}"
done

printf '%s\n' \
  'C2 PRIVATE CHECKPOINT LOAD PASS: strict Clang/GCC/C++, deterministic AOT/runtime, exact reconstruction/rollback, parser/reader failpoints, sanitizers, source/symbol closure, affected regressions'
