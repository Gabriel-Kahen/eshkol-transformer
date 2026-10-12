#!/usr/bin/env bash
# One-use compiled detector control. The ordinary retention proof is separate.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for command in ar cmp diff env git grep nm python3 realpath sha256sum strings timeout xargs; do
  require_command "${command}"
done
[[ -x /usr/bin/time ]] || die "GNU time is required"
[[ $# == 1 && "$1" == /* && ! -e "$1" ]] ||
  die "first-eval detector needs one fresh absolute output directory"
out="$(realpath -m -- "$1")"
mkdir -m 0700 -- "${out}"
trap 'status=$?; printf "%d\n" "$status" >"${out}/gate.exit"; if ((status)); then printf "first-eval evidence retained: %s\n" "$out" >&2; fi' EXIT
git -C "${PROJECT_ROOT}" diff --quiet HEAD -- || die "source tree is dirty"
[[ -z "$(git -C "${PROJECT_ROOT}" ls-files --others --exclude-standard)" ]] ||
  die "source tree has untracked files"
head="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
tree="$(git -C "${PROJECT_ROOT}" rev-parse HEAD^{tree})"
for accepted in 948b8067c1726c2d25ed39cf7eb0a5a29ca8b525 \
    09f7ed52ebd30ec9d2ec2b663a1e6e90e930ebe6; do
  git -C "${PROJECT_ROOT}" merge-base --is-ancestor "${accepted}" "${head}" ||
    die "accepted source/current-main commit absent: ${accepted}"
done
for path in scripts/test-e3-correctness-cases.sh \
    tests/e3_private/correctness_runtime.esk \
    tests/e3_private/parse_correctness_observation.py \
    tests/e3_private/test_correctness_result.py; do
  [[ "$(git -C "${PROJECT_ROOT}" rev-parse "HEAD:${path}")" == \
     "$(git -C "${PROJECT_ROOT}" rev-parse "78604fa3:${path}")" ]] ||
    die "accepted correctness blob differs: ${path}"
done
printf 'head\t%s\ntree\t%s\n' "${head}" "${tree}" >"${out}/source.tsv"
cp "$(eshkol_build_dir)/eshkol-transformer-provenance.tsv" "${out}/toolchain.tsv"
(cd "${PROJECT_ROOT}" && git ls-files -z | xargs -0 -r sha256sum) \
  >"${out}/source.sha256"
python3 -B "${PROJECT_ROOT}/tests/e3_private/retention_first_eval_mutant_source.py" --check \
  >"${out}/derivation.stdout" 2>"${out}/derivation.stderr"

# The accepted numerical and unmodified owner-preservation gates retain their
# own normal/repeat/poison/sanitizer outputs and measured 1024/8192 horizons.
run_logged() {
  local stem=$1 status
  shift
  set +e
  "$@" >"${out}/${stem}.stdout" 2>"${out}/${stem}.stderr"
  status=$?
  set -e
  printf '%d\n' "${status}" >"${out}/${stem}.exit"
  ((status == 0)) || die "${stem} failed; raw streams retained"
}
run_logged correctness "${PROJECT_ROOT}/scripts/test-e3-correctness-cases.sh" \
  "${out}/correctness"
run_logged retention "${PROJECT_ROOT}/scripts/test-e3-complete-call-retention.sh" \
  "${out}/retention"
run_logged native "${PROJECT_ROOT}/scripts/test-e3-first-eval-native-control.sh" \
  "${out}/native"

prefix="${PROJECT_ROOT}/tests/e3_private/retention_first_eval_mutant_package"
for variant in a b sanitized; do
  mode=normal
  [[ "${variant}" != sanitized ]] || mode=sanitize
  artifact="${out}/private-${variant}"
  run_logged "package-${variant}" \
    "${PROJECT_ROOT}/scripts/build-e3-retention-first-eval-mutant.sh" \
    "${artifact}" "${mode}"
  evidence="${artifact}/e3_retention_package.o.evidence"
  for pair in defined_symbols:global-defined public_exports:package-exports \
      public_strings:public-strings source_closure:source-closure \
      native_source_closure:native-source-closure native_objects:native-objects; do
    cmp "${prefix}_${pair%%:*}.txt" "${evidence}/${pair#*:}.txt"
  done
  if [[ "${variant}" != sanitized ]]; then
    cmp "${prefix}_undefined_symbols.txt" "${evidence}/undefined.txt"
  else
    e3_tuple_kind=retention-first-eval-mutant
    source "${PROJECT_ROOT}/scripts/e3-retention-sanitizer-import-policy.sh"
    e3_retention_verify_sanitizer_imports \
      "${prefix}_undefined_symbols.txt" \
      "${evidence}/noninstrumentation-undefined.txt" \
      "${evidence}/sanitizer-undefined.txt"
  fi
  nm -a --defined-only "${artifact}/e3_retention_package.o" |
    grep -E '[[:space:]]et_e3_test_first_eval_corrupt_v1$' >/dev/null ||
    die "mutant helper absent: ${variant}"
  strings -a "${artifact}/e3_retention_package.o" |
    grep -Fx "${out}/.e3-retention-generated/source/e3_d2_dataset.esk" >/dev/null ||
    die "stable generated source path absent: ${variant}"
done
cmp "${out}/private-a/e3_retention_package.o" \
    "${out}/private-b/e3_retention_package.o"
cmp "${out}/private-a/libeshkol_transformer_e3_retention_mutant.a" \
    "${out}/private-b/libeshkol_transformer_e3_retention_mutant.a"
cmp "${out}/private-a/e3_retention_package.o.evidence/private.d" \
    "${out}/private-b/e3_retention_package.o.evidence/private.d"
diff -r "${out}/private-a/e3_retention_package.o.evidence/native-depfiles" \
        "${out}/private-b/e3_retention_package.o.evidence/native-depfiles"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
cat >"${out}/cxx-sanitized" <<WRAPPER
#!/usr/bin/env bash
exec $(printf '%q' "${cxx}") -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
WRAPPER
chmod 0500 "${out}/cxx-sanitized"
for variant in a b sanitized; do
  mkdir "${out}/link-${variant}"
  ar p "${out}/private-${variant}/libeshkol_transformer_e3_retention_mutant.a" \
    e3_retention_package.o >"${out}/link-${variant}/package.o"
  cmp "${out}/private-${variant}/e3_retention_package.o" \
    "${out}/link-${variant}/package.o"
  supplement="${out}/retention/retention_arena.o"
  [[ "${variant}" != sanitized ]] || supplement="${out}/retention/retention_arena_sanitized.o"
  ar rcsD "${out}/link-${variant}/libeshkol_transformer_e3_retention_mutant_test.a" \
    "${out}/link-${variant}/package.o" "${supplement}"
done
cmp "${out}/link-a/libeshkol_transformer_e3_retention_mutant_test.a" \
    "${out}/link-b/libeshkol_transformer_e3_retention_mutant_test.a"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
runtime_timeout="${E3_FIRST_EVAL_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ && \
   "${runtime_timeout}" =~ ^[1-9][0-9]*$ ]] || die "timeouts must be positive"
for variant in a b sanitized; do
  compiler="${cxx}"
  [[ "${variant}" != sanitized ]] || compiler="${out}/cxx-sanitized"
  run_logged "compile-${variant}" \
    env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${compiler}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
    "${runner}" --strict-types --no-stdlib -O 2 \
      -L "${out}/link-${variant}" \
      --lib eshkol_transformer_e3_retention_mutant_test \
      "${PROJECT_ROOT}/tests/e3_private/retention_first_eval_mutant_runtime.esk" \
      -o "${out}/runtime-${variant}"
done
cmp "${out}/runtime-a" "${out}/runtime-b"

corpus="${out}/retention/corpus/packed-single"
[[ -d "${corpus}" ]] || die "retention corpus missing"
for selector in absent-to-present seeded-bit seeded-metadata; do
  gradient=present
  [[ "${selector}" != absent-to-present ]] || gradient=absent
  for profile in normal-a normal-b poisoned-a poisoned-b sanitized; do
    variant=a
    [[ "${profile}" != *-b ]] || variant=b
    [[ "${profile}" != sanitized ]] || variant=sanitized
    poison=0
    [[ "${profile}" != poisoned-* && "${profile}" != sanitized ]] || poison=1
    stem="${selector}-${profile}"
    run_logged "capture-${stem}" \
      env ESHKOL_ARENA_POISON="${poison}" \
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1 \
        /usr/bin/time -f 'elapsed_seconds=%e peak_rss_kib=%M' \
          -o "${out}/${stem}.time" \
        python3 -B -m tests.e3_private.capture_abort \
          --prefix "${out}/${stem}" --timeout "${runtime_timeout}" -- \
          "${out}/runtime-${variant}" "${corpus}" 1024 "${gradient}" "${selector}"
    [[ "$(cat "${out}/${stem}.exit")" == 134 && \
       "$(cat "${out}/${stem}.raw-exit")" == -6 ]] ||
      die "mutant did not produce authentic SIGABRT: ${stem}"
    [[ ! -s "${out}/${stem}.stderr" ]] ||
      die "mutant emitted unexpected diagnostics: ${stem}"
    python3 -B - "${out}/${stem}.stdout" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
if lines != ["E3-RETENTION-FAIL: success-model-image"]:
    raise SystemExit(f"wrong first-eval detector site: {lines!r}")
PY
  done
  for profile in normal-b poisoned-a poisoned-b sanitized; do
    cmp "${out}/${selector}-normal-a.stdout" \
        "${out}/${selector}-${profile}.stdout"
  done
done
printf 'first-eval detector gate passed: %s\n' "${out}"
