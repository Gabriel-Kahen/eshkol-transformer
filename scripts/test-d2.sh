#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

verify_toolchain
for command in ar cmp grep ldd nm python3 rg sha256sum strings timeout; do
  require_command "${command}"
done

d2_tmp="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-transformer-d2.XXXXXX")"
trap 'rm -rf -- "${d2_tmp}"' EXIT
d2_provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
d2_cc="$(tsv_value "${d2_provenance}" cc_path)"
d2_cxx="$(tsv_value "${d2_provenance}" cxx_path)"
d2_runner="$(eshkol_build_dir)/eshkol-run"
d2_dir="$(project_build_dir)/d2"
d2_library="${d2_dir}/libeshkol_transformer_d2_private.a"
i1_library="$(project_build_dir)/i1/libeshkol_transformer_i64.a"
k1_library="$(project_build_dir)/k1/libeshkol_transformer_k1.a"
d2_timeout="${D2_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${d2_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "D2_COMPILER_TIMEOUT_SECONDS must be a positive integer"

[[ -r "${d2_library}" && -r "${d2_dir}/d2_native.o" ]] || \
  die "canonical D2 private archive or object is missing"
[[ "$(ar t "${d2_library}")" == "d2_native.o" ]] || \
  die "canonical D2 private archive has unexpected members"
[[ -r "${i1_library}" && -r "${k1_library}" ]] || \
  die "canonical I1/K1 dependencies are missing"
if nm -g --defined-only --format=posix "${d2_library}" | \
    grep -E '^eshkol_transformer_kernel_provider_v1[[:space:]]'; then
  die "D2 private archive defines the forbidden canonical K1 provider symbol"
fi
nm -g --defined-only --format=posix "${d2_dir}/d2_native.o" | \
  awk '{ print $1 }' >"${d2_tmp}/d2-defined.txt"
nm -u --format=posix "${d2_dir}/d2_native.o" | \
  awk '{ print $1 }' >"${d2_tmp}/d2-undefined.txt"
cmp "${PROJECT_ROOT}/native/d2_native_defined_symbols.txt" \
  "${d2_tmp}/d2-defined.txt"
cmp "${PROJECT_ROOT}/native/d2_native_undefined_symbols.txt" \
  "${d2_tmp}/d2-undefined.txt"

(
  cd -- "${PROJECT_ROOT}"
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v \
    -s tests/d2 -p 'test_*.py'
  for repetition in 1 2; do
    PYTHONDONTWRITEBYTECODE=1 python3 -m tests.d2.generate_batch_fixture \
      --output "${d2_tmp}/fixture-${repetition}.json"
  done
)
cmp "${d2_tmp}/fixture-1.json" "${d2_tmp}/fixture-2.json"
cmp "${d2_tmp}/fixture-1.json" \
  "${PROJECT_ROOT}/tests/d2/fixtures/shifted_batch_v1.json"

compile_semantic_core() {
  local label=$1
  mkdir -p "${d2_tmp}/${label}" "${d2_tmp}/cache-${label}"
  if ! env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
      ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${d2_tmp}/cache-${label}" \
      ESHKOL_CXX_COMPILER="${d2_cxx}" \
      timeout --foreground --signal=TERM --kill-after=5s "${d2_timeout}s" \
      "${d2_runner}" --strict-types --optimize 0 --no-stdlib \
      -I "${PROJECT_ROOT}" "${PROJECT_ROOT}/tests/d2/semantic_core.esk" \
      -o "${d2_tmp}/${label}/d2-semantic" \
      >"${d2_tmp}/${label}/compile.stdout" \
      2>"${d2_tmp}/${label}/compile.stderr"; then
    sed -n '1,240p' "${d2_tmp}/${label}/compile.stderr" >&2
    die "D2 strict AOT semantic-core compilation failed"
  fi
  [[ -x "${d2_tmp}/${label}/d2-semantic" ]] || \
    die "D2 strict AOT executable is missing"
  ! rg -F 'ERROR:' "${d2_tmp}/${label}/compile.stderr" >/dev/null || \
    die "D2 compiler reported ERROR while returning success"
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${d2_tmp}/${label}/d2-semantic" \
    >"${d2_tmp}/${label}/run.stdout" \
    2>"${d2_tmp}/${label}/run.stderr"
  test ! -s "${d2_tmp}/${label}/run.stderr"
  grep -E '^D2 SEMANTIC CORE PASS: [0-9]+ checks$' \
    "${d2_tmp}/${label}/run.stdout" >/dev/null
}

compile_semantic_core a
compile_semantic_core b
cmp "${d2_tmp}/a/d2-semantic" "${d2_tmp}/b/d2-semantic"
cmp "${d2_tmp}/a/run.stdout" "${d2_tmp}/b/run.stdout"
mapfile -t d2_direct_loads < <(
  sed -n 's/^(load "\([^"]*\)").*/\1/p' \
    "${PROJECT_ROOT}/tests/d2/semantic_core.esk"
)
[[ "${#d2_direct_loads[@]}" == 2 &&
   "${d2_direct_loads[0]}" == "internal/c1/lib/c1_sha256.esk" &&
   "${d2_direct_loads[1]}" == "internal/d2/lib/d2_semantic_core.esk" ]] || \
  die "D2 semantic executable source closure changed"
if rg -n '^\(load ' "${PROJECT_ROOT}/internal/d2/lib/d2_semantic_core.esk" \
    >/dev/null; then
  die "D2 semantic core gained an undeclared transitive source load"
fi

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-d2.sh" \
  "${d2_tmp}/native-test" test
d2_cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic
  -fno-common -fstack-protector-all
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native"
)
"${d2_cc}" "${d2_cflags[@]}" -DET_D2_NATIVE_TESTING \
  -DET_I64_TENSOR_TESTING \
  "${PROJECT_ROOT}/tests/d2/test_d2_native.c" \
  "${d2_tmp}/native-test/libeshkol_transformer_d2_private.a" \
  "${PROJECT_ROOT}/native/i64_tensor.c" \
  "${PROJECT_ROOT}/native/kernel_abi.c" -o "${d2_tmp}/test-d2-native"
for repetition in 1 2; do
  timeout --foreground --signal=TERM --kill-after=5s 90s \
    "${d2_tmp}/test-d2-native" \
    >"${d2_tmp}/native-${repetition}.stdout"
done
cmp "${d2_tmp}/native-1.stdout" "${d2_tmp}/native-2.stdout"
grep -E '^D2 native PASS: [0-9]+ ' "${d2_tmp}/native-1.stdout" >/dev/null

"${d2_cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic \
  -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native" \
  "${PROJECT_ROOT}/tests/d2/header_cpp.cpp" \
  "${d2_library}" "${i1_library}" "${k1_library}" \
  -o "${d2_tmp}/header-cpp"
"${d2_tmp}/header-cpp"

/usr/bin/bash "${PROJECT_ROOT}/scripts/build-k1.sh" \
  "${d2_tmp}/sanitized-k1" sanitize
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-i1.sh" \
  "${d2_tmp}/sanitized-i1" sanitize-test
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-d2.sh" \
  "${d2_tmp}/sanitized-d2" sanitize-test
"${d2_cc}" "${d2_cflags[@]}" -DET_D2_NATIVE_TESTING \
  -DET_I64_TENSOR_TESTING \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${PROJECT_ROOT}/tests/d2/test_d2_native.c" \
  "${d2_tmp}/sanitized-d2/libeshkol_transformer_d2_private.a" \
  "${d2_tmp}/sanitized-i1/libeshkol_transformer_i64.a" \
  "${d2_tmp}/sanitized-k1/libeshkol_transformer_k1.a" \
  -o "${d2_tmp}/test-d2-native-sanitized"
ASAN_OPTIONS=detect_leaks="${D2_ASAN_DETECT_LEAKS:-0}":halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  timeout --foreground --signal=TERM --kill-after=5s 90s \
  "${d2_tmp}/test-d2-native-sanitized" >/dev/null

if rg -n -i '\b(import|from) (torch|pytorch|python)|python\.h|py_' \
    "${PROJECT_ROOT}/internal/d2" \
    "${PROJECT_ROOT}/native/d2_native.c" \
    "${PROJECT_ROOT}/native/d2_native.h" >/dev/null; then
  die "D2 production candidate references a Python runtime"
fi
for delivered in "${d2_dir}/d2_native.o" "${d2_tmp}/a/d2-semantic"; do
  if strings -a "${delivered}" | \
      grep -E 'tests/d2/(reference|generate)|torch' >/dev/null || \
      nm -a "${delivered}" | grep -E '(^|[[:space:]])Py_|libpython' >/dev/null || \
      ldd "${delivered}" 2>/dev/null | grep -Ei 'python|torch' >/dev/null; then
    die "D2 delivered candidate contains a development-oracle dependency"
  fi
done

printf 'D2 PASS: carrier-neutral shift/shuffle/cursor semantics, frozen Q0 fixture, private carrier lifetime, determinism, resource, sanitizer, and isolation gates\n'
