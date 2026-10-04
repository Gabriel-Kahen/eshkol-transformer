#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep nm realpath timeout; do require_command "${command}"; done
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
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -fstack-protector-all \
  -fPIC -fvisibility=hidden -fno-common \
  -I "$(eshkol_source_dir)/lib/core" -I "$(eshkol_source_dir)/inc" \
  -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
  -o "${out}/retention_arena.o" \
  >"${out}/arena-compile.stdout" 2>"${out}/arena-compile.stderr"
for cache in a b; do
  mkdir "${out}/link-${cache}"
  ar rcsD "${out}/link-${cache}/libeshkol_transformer_e3_cursor_boundary_test.a" \
    "${out}/package-${cache}/e3_retention_package.o" \
    "${out}/retention_arena.o"
done
cmp "${out}/link-a/libeshkol_transformer_e3_cursor_boundary_test.a" \
    "${out}/link-b/libeshkol_transformer_e3_cursor_boundary_test.a"
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
printf 'E3 cursor boundary accepted: %s\n' "${out}"
