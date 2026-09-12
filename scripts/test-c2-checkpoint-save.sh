#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp diff nm python3 rg sed sort timeout tr; do
  require_command "${command}"
done

cc="${CC:-/usr/bin/clang}"
cxx="${CXX:-/usr/bin/clang++}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-checkpoint-save.XXXXXX")"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    find "${tmp}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,260p' {} \; >&2 || true
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

PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_inspect_fixtures.py" \
  "${tmp}/fixtures-a"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_inspect_fixtures.py" \
  "${tmp}/fixtures-b"
diff -ru "${tmp}/fixtures-a" "${tmp}/fixtures-b"

bridge_sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_codec.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c"
)
build_bridge_driver() {
  local compiler=$1 output=$2
  "${compiler}" "${cflags[@]}" -DET_C2_CHECKPOINT_SAVE_TESTING \
    "${bridge_sources[@]}" \
    "${PROJECT_ROOT}/tests/c2/test_checkpoint_save_bridge.c" -o "${output}"
}
build_bridge_driver "${cc}" "${tmp}/bridge-clang"
"${tmp}/bridge-clang" "${tmp}/fixtures-a/valid.c2" \
  >"${tmp}/bridge-clang.stdout"
grep -Fx 'C2 checkpoint save bridge PASS: 43 checks' \
  "${tmp}/bridge-clang.stdout" >/dev/null

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/c2/checkpoint_save_bridge_header_cpp.cpp" \
  -o "${tmp}/header-clang"
"${tmp}/header-clang"
"${cxx}" -x c++ -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -Wconversion -Wsign-conversion -Wshadow -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" \
  -o "${tmp}/bridge-cxx.o"

if [[ -x /usr/bin/gcc && -x /usr/bin/g++ ]]; then
  build_bridge_driver /usr/bin/gcc "${tmp}/bridge-gcc"
  "${tmp}/bridge-gcc" "${tmp}/fixtures-a/valid.c2" \
    >"${tmp}/bridge-gcc.stdout"
  cmp "${tmp}/bridge-clang.stdout" "${tmp}/bridge-gcc.stdout"
  /usr/bin/g++ -std=c++17 -Wall -Wextra -Werror -Wpedantic \
    -I "${PROJECT_ROOT}/native" \
    "${PROJECT_ROOT}/tests/c2/checkpoint_save_bridge_header_cpp.cpp" \
    -o "${tmp}/header-gcc"
  "${tmp}/header-gcc"
  /usr/bin/g++ -x c++ -std=c++17 -Wall -Wextra -Werror -Wpedantic \
    -Wconversion -Wsign-conversion -Wshadow -I "${PROJECT_ROOT}/native" \
    -c "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" \
    -o "${tmp}/bridge-gxx.o"
fi

"${cc}" "${cflags[@]}" -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" \
  -o "${tmp}/bridge-production.o"
nm -g --defined-only --format=posix "${tmp}/bridge-production.o" | \
  awk '$1 ~ /^et_/ { print $1 }' | LC_ALL=C sort >"${tmp}/symbols"
cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_save_bridge_symbols.txt" \
  "${tmp}/symbols"

while IFS= read -r source; do
  test -f "${PROJECT_ROOT}/${source}" || die "missing native closure file: ${source}"
done <"${PROJECT_ROOT}/native/c2_checkpoint_save_native_source_closure.txt"
"${cc}" -std=c11 -I "${PROJECT_ROOT}/native" -MM \
  "${bridge_sources[@]}" "${PROJECT_ROOT}/native/checkpoint_io.c" | \
  sed -e 's/^[^:]*://' -e 's/\\//g' | tr -s '[:space:]' '\n' | \
  sed "s#^${PROJECT_ROOT}/##" | rg '^native/' | LC_ALL=C sort -u \
  >"${tmp}/native-closure"
LC_ALL=C sort -u \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_native_source_closure.txt" \
  >"${tmp}/expected-native-closure"
cmp "${tmp}/expected-native-closure" "${tmp}/native-closure"

sed -n -E 's/^\(define \((c2-(checkpoint-save|save)-[^ )]+).*/\1/p' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" | LC_ALL=C sort \
  >"${tmp}/definitions"
cmp "${PROJECT_ROOT}/tests/c2/expected/c2_checkpoint_save_definitions.txt" \
  "${tmp}/definitions"
rg -Fx '(define c2-save-stage-components-internal c2-save-stage-components-core)' \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" >/dev/null
if rg '^\(provide|c2-checkpoint-load|c2-[^ ]*capability|[kK]2-' \
    "${PROJECT_ROOT}/native/c2_checkpoint_save_extension.esk" \
    "${PROJECT_ROOT}/native/c2_checkpoint_save_source_closure.txt" >/dev/null; then
  die "private C2 save composed a public/load/capability/K2 surface"
fi

compile_probe() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/probe-${label}"
  (cd "${tmp}/probe-${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/probe-cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 600s "${runner}" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "${tmp}/probe-${label}/save.d" \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t2/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/internal/d2/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" \
        "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_compile_probe.esk" \
        -o "${tmp}/probe-${label}/save.o" \
        >"${tmp}/probe-${label}/compile.stdout" \
        2>"${tmp}/probe-${label}/compile.stderr")
  test ! -s "${tmp}/probe-${label}/compile.stderr"
  sed -e 's/^[^:]*://' -e 's/\\//g' "${tmp}/probe-${label}/save.d" | \
    tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
    sed "s#^${PROJECT_ROOT}/##" | \
    grep -v '^tests/c2/c2_checkpoint_save_compile_probe.esk$' \
    >"${tmp}/probe-${label}/source-closure"
  cmp "${PROJECT_ROOT}/native/c2_checkpoint_save_source_closure.txt" \
    "${tmp}/probe-${label}/source-closure"
  nm -u --format=posix "${tmp}/probe-${label}/save.o" | \
    awk '{ print $1 }' | rg '^et_c2_private_checkpoint_save_' | \
    LC_ALL=C sort -u >"${tmp}/probe-${label}/save-symbols"
  cmp "${PROJECT_ROOT}/tests/c2/expected/checkpoint_save_bridge_symbols.txt" \
    "${tmp}/probe-${label}/save-symbols"
}

compile_probe clang-a "${cxx}"
compile_probe clang-b "${cxx}"
cmp "${tmp}/probe-clang-a/save.o" "${tmp}/probe-clang-b/save.o"
if [[ -x /usr/bin/g++ ]]; then compile_probe gcc /usr/bin/g++; fi

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
runtime_cflags=("${cflags[@]}" -DET_F32_TENSOR_TESTING -DET_O2_TESTING)
"${cc}" "${runtime_cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -DET_C2_I2_MODEL_COPY -c \
  "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" -o "${runtime}/i2.o"
"${cc}" "${runtime_cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE -c \
  "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" -o "${runtime}/o2.o"
"${cc}" "${runtime_cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -DET_P1_TEST_HOOKS=1 -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${runtime}/p1.o"
for source in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor d2_native \
              o2_optimizer c2_x1_canonical c2_checkpoint_codec \
              c2_checkpoint_format; do
  "${cc}" "${runtime_cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${runtime_cflags[@]}" -DET_CHECKPOINT_IO_TESTING -c \
  "${PROJECT_ROOT}/native/checkpoint_io.c" -o "${runtime}/checkpoint_io.o"
"${cc}" "${runtime_cflags[@]}" -DET_C2_CHECKPOINT_SAVE_TESTING -c \
  "${PROJECT_ROOT}/native/c2_checkpoint_save_bridge.c" -o "${runtime}/save.o"
"${cc}" "${runtime_cflags[@]}" -c \
  "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_test_bridge.c" \
  -o "${runtime}/test.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_save.a" "${runtime}"/*.o

(cd "${PROJECT_ROOT}" && PYTHONDONTWRITEBYTECODE=1 python3 \
  -m tests.d2.prepare_public_resources --output "${tmp}/resources-a" && \
  PYTHONDONTWRITEBYTECODE=1 python3 \
  -m tests.d2.prepare_public_resources --output "${tmp}/resources-b")
diff -ru "${tmp}/resources-a" "${tmp}/resources-b"

compile_runtime() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/${label}/output"
  printf old >"${tmp}/${label}/output/existing.c2"
  printf old >"${tmp}/${label}/output/precommit.c2"
  printf old >"${tmp}/${label}/output/postcommit.c2"
  (cd "${tmp}/${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 720s "${runner}" \
        --strict-types --optimize 0 --no-stdlib \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t2/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/internal/d2/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" -L "${runtime}" \
        --lib eshkol_transformer_c2_save \
        "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_runtime.esk" \
        -o "${tmp}/${label}/save" >"${tmp}/${label}/compile.stdout" \
        2>"${tmp}/${label}/compile.stderr")
  test ! -s "${tmp}/${label}/compile.stderr"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    360s "${tmp}/${label}/save" \
      "${PROJECT_ROOT}/tests/t2/fixtures/bpe_tokenizer_v1.tsv" \
      "${tmp}/resources-a" "${tmp}/${label}/output" \
      >"${tmp}/${label}/run.stdout" 2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  grep -E '^C2 PRIVATE CHECKPOINT SAVE PASS: [0-9]+ checks$' \
    "${tmp}/${label}/run.stdout" >/dev/null
}

compile_runtime clang-a "${cxx}"
compile_runtime clang-b "${cxx}"
cmp "${tmp}/clang-a/save" "${tmp}/clang-b/save"
cmp "${tmp}/clang-a/run.stdout" "${tmp}/clang-b/run.stdout"
for image in a.c2 b.c2 postcommit.c2; do
  cmp "${tmp}/clang-a/output/${image}" "${tmp}/clang-b/output/${image}"
done
cmp <(printf old) "${tmp}/clang-a/output/existing.c2"
cmp <(printf old) "${tmp}/clang-a/output/precommit.c2"

if [[ -x /usr/bin/g++ ]]; then
  compile_runtime gcc /usr/bin/g++
  cmp "${tmp}/clang-a/run.stdout" "${tmp}/gcc/run.stdout"
  for image in a.c2 b.c2 postcommit.c2; do
    cmp "${tmp}/clang-a/output/${image}" "${tmp}/gcc/output/${image}"
  done
fi

python3 "${PROJECT_ROOT}/tests/c2/test_checkpoint_save_output.py" \
  "${tmp}/clang-a/output/a.c2" "${tmp}/clang-a/output/b.c2" \
  "${tmp}/clang-a/output/postcommit.c2" "${tmp}/clang-b/output/a.c2"
"${cc}" "${cflags[@]}" "${PROJECT_ROOT}/native/c2_checkpoint_format.c" \
  "${PROJECT_ROOT}/native/c2_x1_canonical.c" \
  "${PROJECT_ROOT}/tests/c2/c2_format_driver.c" -o "${tmp}/format-driver"
"${tmp}/format-driver" "${tmp}/clang-a/output/a.c2" 2 0 \
  >"${tmp}/format-driver.stdout"
grep -E '^0 0 0 1 3 [0-9]+ [0-9]+$' "${tmp}/format-driver.stdout" >/dev/null

"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer -DET_C2_CHECKPOINT_SAVE_TESTING \
  "${bridge_sources[@]}" \
  "${PROJECT_ROOT}/tests/c2/test_checkpoint_save_bridge.c" \
  -o "${tmp}/bridge-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 240s \
    "${tmp}/bridge-san" "${tmp}/fixtures-a/valid.c2" \
    >"${tmp}/bridge-san.stdout"
cmp "${tmp}/bridge-clang.stdout" "${tmp}/bridge-san.stdout"

printf '%s\n' \
  'C2 PRIVATE CHECKPOINT SAVE PASS: strict Clang/GCC/C++, deterministic AOT/artifacts, full parser, failpoints, sanitizers, exact source/symbol closure'
