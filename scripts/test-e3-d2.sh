#!/usr/bin/env bash
# Focused private prerequisite only; not an E3 production aggregate.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH ESHKOL_PATH ESHKOL_LIB_DIR
unset GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CLANG_CONFIG_FILE CCC_OVERRIDE_OPTIONS CCC_CC CCC_CXX
for command in ar cmp diff nm objcopy python3 timeout; do require_command "${command}"; done
mkdir -p "$(project_build_dir)/e3-d2"
work="$(mktemp -d "$(project_build_dir)/e3-d2/run.XXXXXX")"
cleanup() {
  local result=$?
  if [[ "${result}" == 0 && "${E3_D2_KEEP_EVIDENCE:-0}" != 1 ]]; then
    rm -rf -- "${work}"
  else
    printf 'E3 D2 evidence: %s\n' "${work}" >&2
  fi
}
trap cleanup EXIT
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compile_timeout="${E3_D2_COMPILER_TIMEOUT_SECONDS:-600}"
[[ "${compile_timeout}" =~ ^[1-9][0-9]*$ ]] || die "E3_D2_COMPILER_TIMEOUT_SECONDS must be positive"
cd -- "${PROJECT_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
python3 tests/e3_d2/test_source_variant.py
python3 scripts/generate-e3-d2-source.py --output-dir "${work}/variant-a" >"${work}/generator-a.json"
python3 scripts/generate-e3-d2-source.py --output-dir "${work}/variant-b" >"${work}/generator-b.json"
cmp "${work}/generator-a.json" "${work}/generator-b.json"
diff -ru "${work}/variant-a/source" "${work}/variant-b/source"
python3 -m tests.e3_d2.prepare_identity_resources --output "${work}/resources-a"
python3 -m tests.e3_d2.prepare_identity_resources --output "${work}/resources-b"
diff -ru "${work}/resources-a" "${work}/resources-b"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fno-common -fstack-protector-all
       -I "${PROJECT_ROOT}/include" -I "${PROJECT_ROOT}/native")
wrap=(-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=open,--wrap=pread,--wrap=close)
native_test=("${PROJECT_ROOT}/tests/e3_d2/test_native_idle.c"
             "${PROJECT_ROOT}/native/i64_tensor.c" "${PROJECT_ROOT}/native/kernel_abi.c")
"${cc}" "${flags[@]}" -DET_D2_NATIVE_TESTING -DET_I64_TENSOR_TESTING \
  "${native_test[@]}" "${wrap[@]}" -o "${work}/native-idle"
python3 tests/e3_d2/measure_command.py "${work}/native-idle.time.json" \
  "${work}/native-idle" >"${work}/native-idle.stdout"
"${cc}" "${flags[@]}" -DET_D2_NATIVE_TESTING -DET_I64_TENSOR_TESTING \
  -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${native_test[@]}" "${wrap[@]}" -o "${work}/native-idle-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  python3 tests/e3_d2/measure_command.py "${work}/native-idle-sanitized.time.json" \
  "${work}/native-idle-sanitized" >"${work}/native-idle-sanitized.stdout"
cmp "${work}/native-idle.stdout" "${work}/native-idle-sanitized.stdout"
mkdir "${work}/native"
# Original production TU remains separate and is never linked into the variant.
for source in d2_native e3_d2_native; do
  "${cc}" "${flags[@]}" -MMD -MF "${work}/${source}.d" \
    -c "${PROJECT_ROOT}/native/${source}.c" -o "${work}/${source}.o"
  nm -g --defined-only --format=posix "${work}/${source}.o" | \
    awk '{print $1}' | LC_ALL=C sort >"${work}/${source}.symbols"
done
{ cat "${work}/d2_native.symbols"; printf '%s\n' et_e3_d2_dataset_idle_preflight_v1; } | \
  LC_ALL=C sort >"${work}/expected-variant.symbols"
cmp "${work}/expected-variant.symbols" "${work}/e3_d2_native.symbols"
cmp "${PROJECT_ROOT}/native/d2_native_defined_symbols.txt" "${work}/d2_native.symbols"
! grep -E 'test_|_test_' "${work}/e3_d2_native.symbols" || die "test hooks entered production object"
printf '#include "e3_d2_native.h"\nint main(void) { return et_e3_d2_dataset_idle_preflight_v1(0) != 1; }\n' \
  >"${work}/header.cpp"
"${cc}" "${flags[@]}" -c native/i64_tensor.c -o "${work}/header-i64.o"
"${cc}" "${flags[@]}" -c native/kernel_abi.c -o "${work}/header-kernel.o"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -I native -I include \
  "${work}/header.cpp" "${work}/e3_d2_native.o" "${work}/header-i64.o" \
  "${work}/header-kernel.o" -o "${work}/header-cpp"
"${work}/header-cpp"
"${cc}" "${flags[@]}" -x c -c "${work}/header.cpp" -o "${work}/header-c.o"
"${cc}" "${work}/header-c.o" "${work}/e3_d2_native.o" "${work}/header-i64.o" \
  "${work}/header-kernel.o" -o "${work}/header-c"
"${work}/header-c"
if "${cc}" "${work}/header-c.o" "${work}/d2_native.o" "${work}/header-i64.o" \
    "${work}/header-kernel.o" -o "${work}/forbidden-original" \
    >"${work}/forbidden-original.stdout" 2>"${work}/forbidden-original.stderr"; then
  die "original D2 exports E3 idle seam"
fi
grep -F et_e3_d2_dataset_idle_preflight_v1 "${work}/forbidden-original.stderr" >/dev/null
if "${cc}" -r "${work}/d2_native.o" "${work}/e3_d2_native.o" \
    -o "${work}/duplicate.o" >"${work}/duplicate.stdout" 2>"${work}/duplicate.stderr"; then
  die "duplicate original/variant D2 objects linked"
fi
grep -E 'multiple definition|duplicate symbol' "${work}/duplicate.stderr" >/dev/null
for source in p1_identity data_io checkpoint_io kernel_abi t1_i64_shell i64_tensor e3_d2_native; do
  "${cc}" "${flags[@]}" -fPIC -DET_P1_TRUSTED_BUILD=1 -DET_D2_NATIVE_TESTING \
    -MMD -MF "${work}/native/${source}.d" \
    -c "${PROJECT_ROOT}/native/${source}.c" -o "${work}/native/${source}.o"
done
ar rcsD "${work}/native/libe3_d2_test_runtime.a" "${work}/native/"*.o
cat >"${work}/root.esk" <<'ESK'
;;; Test-only canonical T1 prerequisite composition; no E3 frame or public package.
(load "t1_wave1_root.esk")
(load "c1_sha256.esk")
(load "d2_semantic_core.esk")
(load "e3_d2_identity.esk")
(load "e3_d2_dataset.esk")
(load "identity_runtime.esk")
ESK
includes=()
for directory in internal/p1/lib internal/c1/lib internal/t1/lib internal/d2/lib internal/e3/lib src native tests/e3_d2; do
  includes+=(-I "${PROJECT_ROOT}/${directory}")
done
includes+=(-I "${work}/variant-a/source")
# Pinned compiler emits depfiles in compile-only mode, not executable mode.
python3 tests/e3_d2/measure_command.py "${work}/closure.time.json" \
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${work}/cache-closure" ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s "${compile_timeout}s" \
  "${runner}" --strict-types --optimize 0 --no-stdlib "${includes[@]}" \
  --compile-only --emit-depfile "${work}/identity.d" "${work}/root.esk" \
  -o "${work}/identity.o" >"${work}/closure.stdout" 2>"${work}/closure.stderr"
for repetition in a b; do
  python3 tests/e3_d2/measure_command.py "${work}/compile-${repetition}.time.json" \
    env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u ESHKOL_PATH \
    -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${work}/cache-${repetition}" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compile_timeout}s" \
    "${runner}" --strict-types --optimize 0 --no-stdlib "${includes[@]}" \
    -L "${work}/native" --lib e3_d2_test_runtime "${work}/root.esk" \
    -o "${work}/identity-${repetition}" \
    >"${work}/compile-${repetition}.stdout" 2>"${work}/compile-${repetition}.stderr"
  ! grep -F 'ERROR:' "${work}/compile-${repetition}.stderr" || die "E3 D2 compiler reported ERROR"
  ESHKOL_ARENA_POISON=1 python3 tests/e3_d2/measure_command.py "${work}/run-${repetition}.time.json" \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    "${work}/identity-${repetition}" "${work}/resources-${repetition}" \
    >"${work}/identity-${repetition}.stdout" 2>"${work}/identity-${repetition}.stderr"
  test ! -s "${work}/identity-${repetition}.stderr"
  grep -E '^E3 D2 IDENTITY PASS: [0-9]+ genuine T1/generated-D2 checks$' \
    "${work}/identity-${repetition}.stdout" >/dev/null
  python3 tests/e3_d2/check_identity_closure.py "${PROJECT_ROOT}" "${work}" "${repetition}"
done
python3 tests/e3_d2/test_closure_rejections.py "${PROJECT_ROOT}" "${work}"
cmp "${work}/identity-a" "${work}/identity-b"
cmp "${work}/identity-a.stdout" "${work}/identity-b.stdout"
cmp "${work}/resources-a/saved.tsv" "${work}/resources-a/raw.tsv"
# Localize the three prerequisite functions and their source metadata in this
# test executable only. This is not future E3 aggregate export acceptance.
objcopy --localize-symbols="${PROJECT_ROOT}/tests/e3_d2/expected_localized_symbols.txt" \
  "${work}/identity-a" "${work}/identity-localized"
nm --defined-only --format=posix "${work}/identity-localized" >"${work}/localized.symbols"
while IFS= read -r symbol; do
  awk -v symbol="${symbol}" '$1 == symbol && $2 ~ /^[a-z]$/ { found=1 } END { exit !found }' \
    "${work}/localized.symbols" || die "prerequisite symbol is not local: ${symbol}"
done <"${PROJECT_ROOT}/tests/e3_d2/expected_localized_symbols.txt"
ESHKOL_ARENA_POISON=1 python3 tests/e3_d2/measure_command.py "${work}/localized.time.json" \
  timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${work}/identity-localized" "${work}/resources-a" \
  >"${work}/localized.stdout" 2>"${work}/localized.stderr"
test ! -s "${work}/localized.stderr"
cmp "${work}/identity-a.stdout" "${work}/localized.stdout"
cat "${work}/native-idle.stdout" "${work}/identity-a.stdout"
for measurement in "${work}/"*.time.json; do
  printf 'E3 D2 resource %s: ' "$(basename -- "${measurement}")"
  cat "${measurement}"
done
printf 'E3 D2 PREREQUISITE PASS: deterministic source variant, native idle, canonical T1 identity and private closure\n'
