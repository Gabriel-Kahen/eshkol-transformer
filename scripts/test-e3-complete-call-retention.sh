#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env grep python3 timeout wc; do require_command "${command}"; done
[[ -x /usr/bin/time ]] || die "GNU time is required for E3 retention measurements"
[[ $# -le 1 ]] || die "usage: $0 [FRESH_OUTPUT_DIRECTORY]"
out="${1:-$(project_build_dir)/e3-complete-call-retention}"
[[ ! -e "${out}" ]] || die "E3 retention output directory must be fresh: ${out}"
mkdir -p "${out}"

# The accepted private package is rebuilt and checked independently; the
# retention successor may not change its source, symbol or dependency closure.
canonical="${out}/canonical"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-private.sh" "${canonical}" \
  >"${out}/canonical-package.stdout" 2>"${out}/canonical-package.stderr"
/usr/bin/bash "${PROJECT_ROOT}/scripts/test-e3-private-package.sh" "${canonical}" \
  >"${out}/canonical-check.stdout" 2>"${out}/canonical-check.stderr"
for observer in et_e3_test_retention_probe_v1 \
    et_e3_test_model_word_v1 et_d2_exact_read_test_fail_stage_v1 \
    et_f32_test_live_counts_snapshot_v1; do
  if grep -F "${observer}" \
      "${canonical}/e3_private_package.o.evidence/readelf-symbols.txt" >/dev/null; then
    die "retention observer leaked into canonical private package: ${observer}"
  fi
done

for cache in a b; do
  artifact="${out}/private-${cache}"
  /usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-retention-test.sh" \
    "${artifact}" >"${out}/package-${cache}.stdout" \
    2>"${out}/package-${cache}.stderr"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings undefined_symbols:undefined \
      source_closure:source-closure native_source_closure:native-source-closure \
      native_objects:native-objects; do
    expected=${pair%%:*}
    observed=${pair#*:}
    cmp "${PROJECT_ROOT}/tests/e3_private/retention_package_${expected}.txt" \
      "${artifact}/e3_retention_package.o.evidence/${observed}.txt"
  done
  for observer in et_e3_private_retention_dispatch_cabi_v1 \
      et_e3_test_retention_probe_v1 et_e3_test_model_word_count_v1 \
      et_e3_test_model_word_v1 et_e3_test_gradient_metadata_v1 \
      et_e3_test_seed_gradient_v1 et_d2_exact_read_test_fail_stage_v1 \
      et_f32_test_live_counts_snapshot_v1 \
      et_f32_test_retired_counts_snapshot_v1; do
    grep -E "[[:space:]]LOCAL[[:space:]].*[[:space:]]${observer}$" \
      "${artifact}/e3_retention_package.o.evidence/readelf-symbols.txt" \
      >/dev/null || die "retention observer is not private: ${observer}"
  done
done
cmp "${out}/private-a/e3_retention_package.o" \
    "${out}/private-b/e3_retention_package.o"
cmp "${out}/private-a/libeshkol_transformer_e3_retention.a" \
    "${out}/private-b/libeshkol_transformer_e3_retention.a"
# E1B normalizes only its private build root to <E1B_BUILD> when preserving
# raw compiler depfiles. Compare every file and path in both source/C closures.
cmp "${out}/private-a/e3_retention_package.o.evidence/private.d" \
    "${out}/private-b/e3_retention_package.o.evidence/private.d"
diff -r "${out}/private-a/e3_retention_package.o.evidence/native-depfiles" \
        "${out}/private-b/e3_retention_package.o.evidence/native-depfiles"
sanitized="${out}/sanitized-private"
/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-retention-test.sh" \
  "${sanitized}" sanitize \
  >"${out}/sanitized-package.stdout" \
  2>"${out}/sanitized-package.stderr"
for pair in defined_symbols:global-defined public_exports:package-exports \
    public_strings:public-strings source_closure:source-closure \
    native_source_closure:native-source-closure native_objects:native-objects; do
  expected=${pair%%:*}
  observed=${pair#*:}
  cmp "${PROJECT_ROOT}/tests/e3_private/retention_package_${expected}.txt" \
    "${sanitized}/e3_retention_package.o.evidence/${observed}.txt"
done
[[ -s "${sanitized}/e3_retention_package.o.evidence/undefined.txt" ]] || \
  die "E3 sanitized retention object has no undefined inventory"
python3 -m tests.e3_reference.corpus --output "${out}/corpus" \
  >"${out}/corpus.stdout" 2>"${out}/corpus.stderr"
corpus="${out}/corpus/packed-single"
[[ -d "${corpus}" ]] || die "E3 retained corpus is absent"
python3 - "${corpus}" "${out}/corpus-payload.tsv" <<'PY'
from pathlib import Path
import sys
corpus, output = map(Path, sys.argv[1:])
files = sorted(p for p in corpus.rglob("*") if p.is_file())
if not files:
    raise SystemExit("E3 retained corpus has no payload files")
rows = [f"{p.relative_to(corpus).as_posix()}\t{p.stat().st_size}" for p in files]
rows.append(f"TOTAL\t{sum(p.stat().st_size for p in files)}")
output.write_text("\n".join(rows) + "\n")
PY

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
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
for cache in a b; do
  artifact="${out}/private-${cache}"
  mkdir "${out}/link-${cache}"
  ar p "${artifact}/libeshkol_transformer_e3_retention.a" \
    e3_retention_package.o >"${out}/link-${cache}/package.o"
  cmp "${artifact}/e3_retention_package.o" \
    "${out}/link-${cache}/package.o"
  ar rcsD "${out}/link-${cache}/libeshkol_transformer_e3_retention_test.a" \
    "${out}/link-${cache}/package.o" "${out}/retention_arena.o"
done
cmp "${out}/link-a/libeshkol_transformer_e3_retention_test.a" \
    "${out}/link-b/libeshkol_transformer_e3_retention_test.a"
mkdir "${out}/link-sanitized"
ar p "${sanitized}/libeshkol_transformer_e3_retention.a" \
  e3_retention_package.o >"${out}/link-sanitized/package.o"
cmp "${sanitized}/e3_retention_package.o" \
  "${out}/link-sanitized/package.o"
ar rcsD "${out}/link-sanitized/libeshkol_transformer_e3_retention_test.a" \
  "${out}/link-sanitized/package.o" "${out}/retention_arena_sanitized.o"

cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined \
  -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"

compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
runtime_timeout="${E3_COMPLETE_CALL_TIMEOUT_SECONDS:-3600}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ && \
   "${runtime_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "E3 retention timeouts must be positive integers"
for cache in a b; do
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${cache}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/link-${cache}" --lib eshkol_transformer_e3_retention_test \
      "${PROJECT_ROOT}/tests/e3_private/retention_runtime.esk" \
      -o "${out}/runtime-${cache}" \
      >"${out}/compile-${cache}.stdout" \
      2>"${out}/compile-${cache}.stderr"
done
cmp "${out}/runtime-a" "${out}/runtime-b"
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
  -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
  -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
  -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
  -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${out}/cache-sanitized" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
  ESHKOL_CXX_COMPILER="${out}/cxx-sanitized" \
  timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
  "${runner}" --strict-types --no-stdlib -O 2 \
    -L "${out}/link-sanitized" --lib eshkol_transformer_e3_retention_test \
    "${PROJECT_ROOT}/tests/e3_private/retention_runtime.esk" \
    -o "${out}/runtime-sanitized" \
    >"${out}/compile-sanitized.stdout" \
    2>"${out}/compile-sanitized.stderr"

for horizon in 1024 8192; do
  for gradient in absent present; do
    for profile in normal-a normal-b poisoned-a poisoned-b sanitized; do
      poison=0
      [[ "${profile}" == poisoned-* || "${profile}" == sanitized ]] && poison=1
      executable="${out}/runtime-a"
      [[ "${profile}" != *-b ]] || executable="${out}/runtime-b"
      [[ "${profile}" != sanitized ]] || executable="${out}/runtime-sanitized"
      ESHKOL_ARENA_POISON="${poison}" \
        E3_RETENTION_RSS_PATH="${out}/${horizon}-${gradient}-${profile}.rss" \
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 \
        /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
          -o "${out}/${horizon}-${gradient}-${profile}.time" \
        timeout --foreground --signal=TERM --kill-after=5s \
          "${runtime_timeout}s" "${executable}" \
          "${corpus}" "${horizon}" "${gradient}" \
          >"${out}/${horizon}-${gradient}-${profile}.stdout" \
          2>"${out}/${horizon}-${gradient}-${profile}.stderr"
      grep -E "^E3-COMPLETE-CALL-RETENTION-PASS horizon=${horizon} gradient=${gradient} " \
        "${out}/${horizon}-${gradient}-${profile}.stdout" >/dev/null
      grep -E '^rss_before_kib=[0-9]+ rss_after_kib=[0-9]+$' \
        "${out}/${horizon}-${gradient}-${profile}.rss" >/dev/null
      [[ "$(wc -l <"${out}/${horizon}-${gradient}-${profile}.stdout")" == 1 ]] || \
        die "E3 retention runtime wrote unexpected stdout: ${horizon}/${gradient}/${profile}"
      [[ ! -s "${out}/${horizon}-${gradient}-${profile}.stderr" ]] || \
        die "E3 retention runtime wrote stderr: ${horizon}/${gradient}/${profile}"
    done
    for profile in normal-b poisoned-a poisoned-b sanitized; do
      cmp "${out}/${horizon}-${gradient}-normal-a.stdout" \
        "${out}/${horizon}-${gradient}-${profile}.stdout"
    done
  done
done
printf 'E3 complete-call retention passed: %s\n' "${out}"
