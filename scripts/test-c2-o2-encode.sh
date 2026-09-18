#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar awk cmp nm python3 rg sed sort timeout tr; do
  require_command "${command}"
done

cc=
cxx=
resolve_provenance_compilers cc cxx \
  "${CC:-$(lock_value supported_cc)}" "${CXX:-$(lock_value supported_cxx)}"
runner="$(eshkol_build_dir)/eshkol-run"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-o2-encode.XXXXXX")"
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
  -I "${PROJECT_ROOT}/native" -DET_F32_TENSOR_TESTING -DET_O2_TESTING
)

runtime="${tmp}/runtime"
mkdir -p "${runtime}"
"${cc}" "${cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE -MMD -MF "${runtime}/o2_bridge.d" \
  -c "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
  -o "${runtime}/o2_bridge.o"
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${runtime}/i2_bridge.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" -o "${runtime}/p1_identity.o"
for source in data_io checkpoint_io kernel_abi i64_tensor t1_i64_shell \
              f32_tensor o2_optimizer; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime}/${source}.o"
done
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/c2/c2_o2_encode_test_bridge.c" \
  -o "${runtime}/test_bridge.o"
ar rcsD "${runtime}/libeshkol_transformer_c2_o2_encode.a" "${runtime}"/*.o

nm -g --defined-only --format=posix "${runtime}/o2_bridge.o" | \
  awk '$1 ~ /^et_c2_/ { print $1 }' | LC_ALL=C sort \
  >"${tmp}/native-symbols"
cmp "${PROJECT_ROOT}/tests/c2/expected/c2_o2_encode_native_symbols.txt" \
  "${tmp}/native-symbols"
"${cc}" "${cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
  -o "${tmp}/o2-without-c2.o"
if nm -g --defined-only --format=posix "${tmp}/o2-without-c2.o" | \
    rg '^et_c2_[^[:space:]]+[[:space:]]' >/dev/null; then
  die "ordinary O2 bridge exposed private C2 reconstruction/copy authority"
fi

sed -e 's/^[^:]*://' -e 's/\\//g' "${runtime}/o2_bridge.d" | \
  tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" >"${tmp}/native-source-closure"
cmp "${PROJECT_ROOT}/native/c2_o2_encode_native_source_closure.txt" \
  "${tmp}/native-source-closure"

sed -n -E 's/^\(define \(?(c2-o2-[^ )]+).*/\1/p' \
  "${PROJECT_ROOT}/native/c2_o2_encode_extension.esk" | LC_ALL=C sort \
  >"${tmp}/adapter-definitions"
cmp "${PROJECT_ROOT}/tests/c2/expected/c2_o2_encode_definitions.txt" \
  "${tmp}/adapter-definitions"
if rg '^\(provide|k2|K2|optimizer-create|optimizer-step!' \
    "${PROJECT_ROOT}/native/c2_o2_encode_extension.esk" \
    "${PROJECT_ROOT}/native/c2_o2_encode_source_closure.txt" >/dev/null; then
  die "private C2 O2 staging composed a public facade or K2 authority"
fi
{
  cat "${PROJECT_ROOT}/native/o2_wave2_source_closure.txt"
  printf '%s\n' native/c2_o2_reconstruct_extension.esk \
    native/c2_o2_encode_extension.esk
} >"${tmp}/expected-source-closure"
cmp "${PROJECT_ROOT}/native/c2_o2_encode_source_closure.txt" \
  "${tmp}/expected-source-closure"

compile_probe() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/probe-${label}"
  (cd "${tmp}/probe-${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-probe-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 420s "${runner}" \
        --strict-types --optimize 0 --no-stdlib --emit-object \
        --emit-depfile "${tmp}/probe-${label}/probe.d" \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" \
        "${PROJECT_ROOT}/tests/c2/c2_o2_encode_compile_probe.esk" \
        -o "${tmp}/probe-${label}/probe.o" \
        >"${tmp}/probe-${label}/compile.stdout" \
        2>"${tmp}/probe-${label}/compile.stderr")
  test ! -s "${tmp}/probe-${label}/compile.stderr"
  sed -e 's/^[^:]*://' -e 's/\\//g' "${tmp}/probe-${label}/probe.d" | \
    tr -s '[:space:]' '\n' | grep -F "${PROJECT_ROOT}/" | \
    sed "s#^${PROJECT_ROOT}/##" | \
    grep -v '^tests/c2/c2_o2_encode_compile_probe.esk$' \
    >"${tmp}/probe-${label}/source-closure"
  cmp "${PROJECT_ROOT}/native/c2_o2_encode_source_closure.txt" \
    "${tmp}/probe-${label}/source-closure"
}

compile_runtime() {
  local label=$1 compiler=$2
  mkdir -p "${tmp}/${label}"
  (cd "${tmp}/${label}" &&
    env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
      XDG_CACHE_HOME="${tmp}/cache-${label}" \
      ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
      timeout --foreground --signal=TERM --kill-after=5s 480s "${runner}" \
        --strict-types --optimize 0 --no-stdlib \
        -I "${PROJECT_ROOT}/internal/p1/lib" \
        -I "${PROJECT_ROOT}/internal/c1/lib" \
        -I "${PROJECT_ROOT}/internal/t1/lib" \
        -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
        -I "${PROJECT_ROOT}/native" -L "${runtime}" \
        --lib eshkol_transformer_c2_o2_encode \
        "${PROJECT_ROOT}/tests/c2/c2_o2_encode_runtime.esk" \
        -o "${tmp}/${label}/runtime" \
        >"${tmp}/${label}/compile.stdout" \
        2>"${tmp}/${label}/compile.stderr")
  test ! -s "${tmp}/${label}/compile.stderr"
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s \
    240s "${tmp}/${label}/runtime" \
      "${tmp}/${label}/metadata.bin" "${tmp}/${label}/payload.bin" \
      >"${tmp}/${label}/run.stdout" 2>"${tmp}/${label}/run.stderr"
  test ! -s "${tmp}/${label}/run.stderr"
  grep -E '^C2 O2 ENCODE PASS: [0-9]+ checks$' \
    "${tmp}/${label}/run.stdout" >/dev/null
}

compile_probe clang-a "${cxx}"
compile_probe clang-b "${cxx}"
cmp "${tmp}/probe-clang-a/probe.o" "${tmp}/probe-clang-b/probe.o"
compile_runtime clang-a "${cxx}"
compile_runtime clang-b "${cxx}"
cmp "${tmp}/clang-a/runtime" "${tmp}/clang-b/runtime"
cmp "${tmp}/clang-a/run.stdout" "${tmp}/clang-b/run.stdout"
cmp "${tmp}/clang-a/metadata.bin" "${tmp}/clang-b/metadata.bin"
cmp "${tmp}/clang-a/payload.bin" "${tmp}/clang-b/payload.bin"

if [[ -x /usr/bin/gcc && -x /usr/bin/g++ ]]; then
  /usr/bin/gcc "${cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
    -DET_C2_O2_RECONSTRUCT_BRIDGE \
    -c "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
    -o "${tmp}/gcc-o2-bridge.o"
  /usr/bin/gcc "${cflags[@]}" \
    -c "${PROJECT_ROOT}/tests/c2/c2_o2_encode_test_bridge.c" \
    -o "${tmp}/gcc-test-bridge.o"
  compile_probe gcc /usr/bin/g++
  compile_runtime gcc /usr/bin/g++
  cmp "${tmp}/clang-a/run.stdout" "${tmp}/gcc/run.stdout"
  cmp "${tmp}/clang-a/metadata.bin" "${tmp}/gcc/metadata.bin"
  cmp "${tmp}/clang-a/payload.bin" "${tmp}/gcc/payload.bin"
fi

PYTHONPATH="${PROJECT_ROOT}/tests/c2" python3 \
  "${PROJECT_ROOT}/tests/c2/test_o2_encode_staging.py" \
  "${tmp}/reference-metadata.bin" "${tmp}/reference-payload.bin" \
  "${tmp}/reference-input.c2"
cmp "${tmp}/reference-metadata.bin" "${tmp}/clang-a/metadata.bin"
cmp "${tmp}/reference-payload.bin" "${tmp}/clang-a/payload.bin"

codec_sources=(
  "${PROJECT_ROOT}/native/c2_checkpoint_codec.c"
  "${PROJECT_ROOT}/native/c2_checkpoint_format.c"
  "${PROJECT_ROOT}/native/c2_x1_canonical.c"
  "${PROJECT_ROOT}/tests/c2/c2_codec_driver.c"
)
"${cc}" "${cflags[@]}" "${codec_sources[@]}" -o "${tmp}/codec-driver"
"${tmp}/codec-driver" "${tmp}/reference-input.c2" \
  "${tmp}/round-trip.c2" 0 >"${tmp}/codec.stdout"

"${cc}" "${cflags[@]}" -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer "${codec_sources[@]}" -o "${tmp}/codec-driver-san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "${tmp}/codec-driver-san" "${tmp}/reference-input.c2" \
    "${tmp}/round-trip-san.c2" 0 >"${tmp}/codec-san.stdout"
cmp "${tmp}/round-trip.c2" "${tmp}/round-trip-san.c2"

printf 'C2 O2 ENCODE PASS: exact Python wire bytes, live-state no-borrow staging, aliases/groups, -0 bits, strict Clang/GCC AOT, parser/codec round-trip, flat controls, sanitizers\n'
