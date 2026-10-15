#!/usr/bin/env bash
# Deferred heavy gate. Never run as part of the N5 source-only handoff.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env nm python3 timeout; do require_command "${command}"; done
[[ $# -eq 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "${out}" ]] || die "N5 output directory must be fresh"
mkdir -p "${out}"
python3 -B -m tests.e3_private.n5_corpus --output "${out}/corpus-a"
python3 -B -m tests.e3_private.n5_corpus --output "${out}/corpus-b"
diff -r "${out}/corpus-a" "${out}/corpus-b"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
"${cxx}" -std=c++17 -Wall -Wextra -Werror -fstack-protector-all \
  -fPIC -fvisibility=hidden -fno-common \
  -I "$(eshkol_source_dir)/lib/core" -I "$(eshkol_source_dir)/inc" \
  -c "${PROJECT_ROOT}/tests/e3_private/retention_arena.cpp" \
  -o "${out}/retention_arena.o"
for mode in normal gte; do
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-n5-cap-test.sh" \
    "${out}/private-${mode}" "${mode}" \
    >"${out}/build-${mode}.stdout" 2>"${out}/build-${mode}.stderr"
  closure="${PROJECT_ROOT}/tests/e3_private/fault_package_source_closure.txt"
  [[ "${mode}" != gte ]] || \
    closure="${PROJECT_ROOT}/tests/e3_private/fault_n5_cap_mutant_source_closure.txt"
  cmp "${closure}" \
    "${out}/private-${mode}/e3_fault_package.o.evidence/source-closure.txt"
  mkdir "${out}/link-${mode}"
  ar rcsD "${out}/link-${mode}/libeshkol_transformer_e3_n5.a" \
    "${out}/private-${mode}/e3_fault_package.o" \
    "${out}/retention_arena.o"
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u GCC_EXEC_PREFIX -u COMPILER_PATH \
    -u LIBRARY_PATH -u CLANG_CONFIG_FILE -u CCC_OVERRIDE_OPTIONS \
    -u CCC_CC -u CCC_CXX -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR \
    ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME="${out}/cache-${mode}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${E3_COMPILER_TIMEOUT_SECONDS:-900}s" \
      "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/link-${mode}" --lib eshkol_transformer_e3_n5 \
      "${PROJECT_ROOT}/tests/e3_private/n5_cap_runtime.esk" \
      -o "${out}/runtime-${mode}" \
      >"${out}/compile-${mode}.stdout" 2>"${out}/compile-${mode}.stderr"
  if nm -u "${out}/runtime-${mode}" | \
      grep -E '[[:space:]]U et_e3_fault_run_v1(@|$)' >/dev/null; then
    die "N5 private CABI remained unresolved"
  fi
  for gradient in absent present; do
    timeout --foreground --signal=TERM --kill-after=5s \
      "${E3_N5_RUNTIME_TIMEOUT_SECONDS:-3600}s" \
      "${out}/runtime-${mode}" "${out}/corpus-a" "${gradient}" "${mode}" \
      >"${out}/${mode}-${gradient}.stdout" \
      2>"${out}/${mode}-${gradient}.stderr"
    result=success
    [[ "${mode}" != gte ]] || result=unsupported
    printf 'E3-N5-CAP-PASS gradient=%s mode=%s one_over=8193 boundary=8192 first_error=29 boundary_result=%s\n' \
      "${gradient}" "${mode}" "${result}" | \
      cmp - "${out}/${mode}-${gradient}.stdout"
    [[ ! -s "${out}/${mode}-${gradient}.stderr" ]] || \
      die "N5 runtime wrote stderr"
  done
done
printf 'E3-N5-COMPILED-PASS output=%s\n' "${out}"
