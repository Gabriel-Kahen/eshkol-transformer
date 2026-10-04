#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cat chmod cmp diff env grep nm realpath sort timeout wc; do
  require_command "${command}"
done
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="$(realpath -m -- "${1:-$(project_build_dir)/e3-cursor-boundary}")"
[[ ! -e "${out}" ]] || die "E3 cursor boundary output must be fresh"
mkdir -p "${out}"
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] || \
  die "E3 cursor boundary requires absolute pinned Q0_PYTHON"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] || \
  die "E3 cursor boundary oracle versions are not pinned"
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" -B -m \
  unittest tests.e3_reference.test_cursor_boundary \
  >"${out}/reference.stdout" 2>"${out}/reference.stderr"
"${q0_python}" -B - "${out}/corpus-a" <<'PY'
from pathlib import Path
import sys
from tests.e3_reference.cursor_boundary import materialize
materialize(Path(sys.argv[1]))
PY
"${q0_python}" -B - "${out}/corpus-b" <<'PY'
from pathlib import Path
import sys
from tests.e3_reference.cursor_boundary import materialize
materialize(Path(sys.argv[1]))
PY
diff -r "${out}/corpus-a" "${out}/corpus-b"
prefix="${PROJECT_ROOT}/tests/e3_private/cursor_boundary_package"
for cache in a b; do
  "${PROJECT_ROOT}/scripts/build-e3-cursor-boundary-test.sh" \
    "${out}/package-${cache}" >"${out}/package-${cache}.stdout" \
    2>"${out}/package-${cache}.stderr"
  evidence="${out}/package-${cache}/e3_retention_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings undefined_symbols:undefined \
      source_closure:source-closure native_source_closure:native-source-closure \
      native_objects:native-objects; do
    expected=${pair%%:*}; observed=${pair#*:}
    cmp "${prefix}_${expected}.txt" "${evidence}/${observed}.txt"
  done
  for symbol in et_e3_private_cursor_boundary_dispatch_cabi_v1 \
      et_e3_test_retention_probe_v1 et_e3_test_model_word_v1; do
    grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${symbol}$" \
      "${evidence}/readelf-symbols.txt" >/dev/null || \
      die "cursor boundary observer is not private: ${symbol}"
  done
done
cmp "${out}/package-a/e3_retention_package.o" \
    "${out}/package-b/e3_retention_package.o"
cmp "${out}/package-a/e3_retention_package.o.evidence/private.d" \
    "${out}/package-b/e3_retention_package.o.evidence/private.d"
diff -r "${out}/package-a/e3_retention_package.o.evidence/native-depfiles" \
        "${out}/package-b/e3_retention_package.o.evidence/native-depfiles"
sanitized="${out}/package-sanitized"
"${PROJECT_ROOT}/scripts/build-e3-cursor-boundary-test.sh" \
  "${sanitized}" sanitize >"${out}/package-sanitized.stdout" \
  2>"${out}/package-sanitized.stderr"
evidence="${sanitized}/e3_retention_package.o.evidence"
for pair in defined_symbols:global-defined public_exports:package-exports \
    public_strings:public-strings source_closure:source-closure \
    native_source_closure:native-source-closure native_objects:native-objects; do
  expected=${pair%%:*}; observed=${pair#*:}
  cmp "${prefix}_${expected}.txt" "${evidence}/${observed}.txt"
done
for symbol in et_e3_private_cursor_boundary_dispatch_cabi_v1 \
    et_e3_test_retention_probe_v1 et_e3_test_model_word_v1; do
  grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${symbol}$" \
    "${evidence}/readelf-symbols.txt" >/dev/null || \
    die "sanitized cursor boundary observer is not private: ${symbol}"
done
source "${PROJECT_ROOT}/scripts/e3-retention-sanitizer-import-policy.sh"
e3_retention_verify_sanitizer_imports \
  "${prefix}_undefined_symbols.txt" \
  "${evidence}/noninstrumentation-undefined.txt" \
  "${evidence}/sanitizer-undefined.txt"
cat "${PROJECT_ROOT}/tests/e3_private/retention_package_sanitized_noninstrumentation_undefined_symbols.txt" \
    "${PROJECT_ROOT}/tests/e3_private/retention_package_sanitized_instrumentation_undefined_symbols.txt" |
  LC_ALL=C sort -u >"${out}/expected-sanitized-undefined.txt"
cmp "${out}/expected-sanitized-undefined.txt" "${evidence}/undefined.txt"
[[ "$(wc -l <"${out}/expected-sanitized-undefined.txt")" == 226 ]] ||
  die "sanitized cursor boundary imports differ from the closed tuple"
cmp "${out}/package-a/e3_retention_package.o.evidence/private.d" \
    "${evidence}/private.d"
diff -r "${out}/package-a/e3_retention_package.o.evidence/native-depfiles" \
        "${evidence}/native-depfiles"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -fstack-protector-all \
  -fPIC -fvisibility=hidden -fno-common \
  -I "$(eshkol_source_dir)/lib/core" -I "$(eshkol_source_dir)/inc" \
  -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
  -o "${out}/retention_arena.o" \
  >"${out}/arena-compile.stdout" 2>"${out}/arena-compile.stderr"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -fstack-protector-all \
  -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -fPIC -fvisibility=hidden -fno-common \
  -I "$(eshkol_source_dir)/lib/core" -I "$(eshkol_source_dir)/inc" \
  -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
  -o "${out}/retention_arena_sanitized.o" \
  >"${out}/arena-sanitized-compile.stdout" \
  2>"${out}/arena-sanitized-compile.stderr"
for supplement in "${out}/retention_arena.o" \
    "${out}/retention_arena_sanitized.o"; do
  nm -g --defined-only "${supplement}" |
    grep -E '[[:space:]]T et_e3_retention_region_used_v1$' >/dev/null ||
    die "cursor boundary supplement lacks its private region observer"
done
for cache in a b; do
  mkdir "${out}/link-${cache}"
  ar rcsD "${out}/link-${cache}/libeshkol_transformer_e3_cursor_boundary_test.a" \
    "${out}/package-${cache}/e3_retention_package.o" \
    "${out}/retention_arena.o"
done
cmp "${out}/link-a/libeshkol_transformer_e3_cursor_boundary_test.a" \
    "${out}/link-b/libeshkol_transformer_e3_cursor_boundary_test.a"
mkdir "${out}/link-sanitized"
ar p "${sanitized}/libeshkol_transformer_e3_cursor_boundary.a" \
  e3_retention_package.o >"${out}/link-sanitized/package.o"
cmp "${sanitized}/e3_retention_package.o" "${out}/link-sanitized/package.o"
ar rcsD "${out}/link-sanitized/libeshkol_transformer_e3_cursor_boundary_test.a" \
  "${out}/link-sanitized/package.o" "${out}/retention_arena_sanitized.o"
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined \
  -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E3 cursor boundary compiler timeout must be positive"
runner="$(eshkol_build_dir)/eshkol-run"
for cache in a b; do
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
    -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES \
    -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX -u ESHKOL_PATH \
    -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${cache}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/link-${cache}" \
      --lib eshkol_transformer_e3_cursor_boundary_test \
      "${PROJECT_ROOT}/tests/e3_private/cursor_boundary_runtime.esk" \
      -o "${out}/runtime-${cache}" \
      >"${out}/compile-${cache}.stdout" \
      2>"${out}/compile-${cache}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
  -u OBJC_INCLUDE_PATH -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES \
  -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
  -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX -u ESHKOL_PATH \
  -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${out}/cache-sanitized" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  ESHKOL_CXX_COMPILER="${out}/cxx-sanitized" \
  timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
  "${runner}" --strict-types --no-stdlib -O 2 \
    -L "${out}/link-sanitized" \
    --lib eshkol_transformer_e3_cursor_boundary_test \
    "${PROJECT_ROOT}/tests/e3_private/cursor_boundary_runtime.esk" \
    -o "${out}/runtime-sanitized" \
    >"${out}/compile-sanitized.stdout" \
    2>"${out}/compile-sanitized.stderr"
for executable in "${out}/runtime-a" "${out}/runtime-b" \
    "${out}/runtime-sanitized"; do
  if nm -u "${executable}" |
      grep -E '[[:space:]]U et_e3_(private_cursor_boundary_dispatch|test_retention_probe|test_model_word)_.*(@|$)' >/dev/null; then
    die "cursor boundary private observer remained unresolved after final AOT link"
  fi
done
for cache in a b; do
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 900s \
    "${out}/runtime-${cache}" \
    "${out}/corpus-a/empty" "${out}/corpus-a/singleton" \
    "${out}/corpus-a/entry-eos" "${out}/corpus-a/final-suffix" \
    "${out}/corpus-a/one-over" \
    >"${out}/run-${cache}.stdout" 2>"${out}/run-${cache}.stderr"
  printf 'E3-CURSOR-BOUNDARY-PASS\n' | cmp - "${out}/run-${cache}.stdout"
done
cmp "${out}/run-a.stdout" "${out}/run-b.stdout"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 ESHKOL_ARENA_POISON=1 \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${out}/runtime-sanitized" \
  "${out}/corpus-a/empty" "${out}/corpus-a/singleton" \
  "${out}/corpus-a/entry-eos" "${out}/corpus-a/final-suffix" \
  "${out}/corpus-a/one-over" \
  >"${out}/run-sanitized.stdout" 2>"${out}/run-sanitized.stderr"
printf 'E3-CURSOR-BOUNDARY-PASS\n' | cmp - "${out}/run-sanitized.stdout"
[[ ! -s "${out}/run-sanitized.stderr" ]] ||
  die "sanitized cursor boundary runtime emitted diagnostics"
cmp "${out}/run-a.stdout" "${out}/run-sanitized.stdout"
printf 'E3 cursor boundary accepted: %s\n' "${out}"
