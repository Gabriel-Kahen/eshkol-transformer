#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
verify_toolchain
for tool in cmp diff env llvm-objdump nm python3 rg timeout; do
  require_command "${tool}"
done
[[ $# -eq 1 ]] || die "usage: $0 FRESH_OUTPUT_DIRECTORY"
out="$(realpath -m -- "$1")"
[[ ! -e "${out}" && ! -L "${out}" ]] || die "output must be fresh"
mkdir -p "${out}"
trap 'exit 143' TERM
trap 'exit 130' INT
q0_python="${Q0_PYTHON:-}"
[[ "${q0_python}" == /* && -x "${q0_python}" ]] ||
  die "absolute pinned Q0_PYTHON is required"
versions="$(ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  "${q0_python}" - <<'PY'
import platform, torch
print(platform.python_version())
print(torch.__version__)
PY
)"
[[ "${versions}" == $'3.14.6\n2.13.0+cpu' ]] ||
  die "I2 guard oracle versions differ from the locked development tuple"
for repeat in a b; do
  ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
    -m tests.e3_private.zero_control_reference --output "${out}/oracle-${repeat}" \
    >"${out}/oracle-${repeat}.stdout" 2>"${out}/oracle-${repeat}.stderr"
done
diff -r "${out}/oracle-a" "${out}/oracle-b"

provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cxx="$(tsv_value "${provenance}" cxx_path)"
runner="$(eshkol_build_dir)/eshkol-run"
compiler_timeout="${E3_COMPILER_TIMEOUT_SECONDS:-900}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "invalid compiler timeout"
printf '#!/usr/bin/env bash\nexec %q -fsanitize=address,undefined -fno-omit-frame-pointer "$@"\n' \
  "${cxx}" >"${out}/sanitizer-cxx"
chmod 700 "${out}/sanitizer-cxx"

compile_runtime() {
  local variant=$1 kind=$2 optimize=$3 compiler=$4
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
    -u DEPENDENCIES_OUTPUT -u SUNPRO_DEPENDENCIES -u GCC_EXEC_PREFIX \
    -u COMPILER_PATH -u LIBRARY_PATH -u CLANG_CONFIG_FILE \
    -u CCC_OVERRIDE_OPTIONS -u CCC_CC -u CCC_CXX \
    -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${out}/cache-${variant}-${kind}" \
    ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" ESHKOL_CXX_COMPILER="${compiler}" \
    timeout --foreground --signal=TERM --kill-after=5s "${compiler_timeout}s" \
      "${runner}" --strict-types --no-stdlib -O "${optimize}" \
      -L "${out}/package-${variant}" --lib eshkol_transformer_e3_zero_control \
      "${PROJECT_ROOT}/tests/e3_private/${kind}_runtime.esk" \
      -o "${out}/${kind}-runtime-${variant}" \
      >"${out}/${kind}-compile-${variant}.stdout" \
      2>"${out}/${kind}-compile-${variant}.stderr"
  [[ ! -s "${out}/${kind}-compile-${variant}.stderr" ]] ||
    die "${variant}/${kind} compiled with diagnostics"
}

"${PROJECT_ROOT}/scripts/build-e3-zero-control-test.sh" \
  "${out}/package-control" normal \
  >"${out}/package-control.stdout" 2>"${out}/package-control.stderr"
compile_runtime control zero_control 2 "${cxx}"
ESHKOL_ARENA_POISON=0 \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${out}/zero_control-runtime-control" \
  "${out}/oracle-a/all-correct" all-correct absent \
  >"${out}/control.stdout" 2>"${out}/control.stderr"
rg -Fx 'E3-ZERO-CONTROL-RUNTIME-PASS' "${out}/control.stdout" >/dev/null ||
  die "normal real-model control did not complete"
[[ ! -s "${out}/control.stderr" ]] || die "normal control emitted diagnostics"
python3 -m tests.e3_private.parse_zero_control_observation \
  --stdout "${out}/control.stdout" \
  --observed "${out}/control-observed.json" \
  --roles "${out}/control-roles.tsv"
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${q0_python}" \
  -m tests.e3_native_parity.verify_observation \
    --reference "${out}/oracle-a/reference.json" --case all-correct \
    --observed "${out}/control-observed.json" \
    --roles "${out}/control-roles.tsv" \
    --transcript "${out}/control-transcript.json" \
    >"${out}/control-verify.stdout" 2>"${out}/control-verify.stderr"
[[ ! -s "${out}/control-verify.stderr" ]] ||
  die "normal control oracle verification emitted diagnostics"

for length in 1 3; do
  for repeat in a b sanitize; do
    variant="${length}-${repeat}"
    mode=normal optimizer=2 compiler="${cxx}"
    if [[ "${repeat}" == sanitize ]]; then
      mode=sanitize optimizer=1 compiler="${out}/sanitizer-cxx"
    fi
    E3_I2_POSTPREPARE_GUARD_LENGTH="${length}" \
      "${PROJECT_ROOT}/scripts/build-e3-zero-control-test.sh" \
        "${out}/package-${variant}" "${mode}" \
        >"${out}/package-${variant}.stdout" \
        2>"${out}/package-${variant}.stderr"
    object="${out}/package-${variant}/e3_zero_control_package.o"
    evidence="${object}.evidence"
    cmp "${PROJECT_ROOT}/tests/e3_private/i2_guard_package_native_source_closure.txt" \
      "${evidence}/native-source-closure.txt"
    for symbol in et_i2_private_copy_builder_prepare_v1 \
        et_i2_private_copy_builder_abort_v1 \
        arena_allocate_vector_with_header; do
      nm -u --format=posix "${evidence}/private.prelink.o" | \
        awk '{print $1}' | rg -Fx "${symbol}" >/dev/null ||
        die "private object omitted prelink ${symbol} reference"
    done
    for symbol in et_i2_private_copy_builder_prepare_v1 \
        et_i2_private_copy_builder_abort_v1 \
        et_i2_private_copy_builder_commit_v1 \
        arena_allocate_vector_with_header; do
      nm -g --defined-only --format=posix "${evidence}/combined.raw.o" | \
        awk '{print $1}' | rg -Fx "__wrap_${symbol}" >/dev/null ||
        die "partial link omitted real ${symbol} wrapper definition"
    done
    if nm -u --format=posix "${evidence}/combined.raw.o" | \
        awk '{print $1}' | rg '^__real_' >/dev/null; then
      die "partial link retained an unresolved real-symbol alias"
    fi
    llvm-objdump -dr \
      --disassemble-symbols=i2-prepare-load-guarded__eshkol_internal_abi \
      "${evidence}/combined.raw.o" >"${out}/link-${variant}.txt"
    for wrapped in et_i2_private_copy_builder_prepare_v1 \
        et_i2_private_copy_builder_abort_v1 \
        arena_allocate_vector_with_header; do
      rg -F "__wrap_${wrapped}" "${out}/link-${variant}.txt" >/dev/null ||
        die "partial link omitted wrapped production call ${wrapped}"
    done
    compile_runtime "${variant}" zero_control_negative \
      "${optimizer}" "${compiler}"
    for gradient in absent present; do
      ESHKOL_ARENA_POISON=0 \
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        LSAN_OPTIONS=exitcode=23 \
        timeout --foreground --signal=TERM --kill-after=5s 900s \
        "${out}/zero_control_negative-runtime-${variant}" \
        "${out}/oracle-a/all-correct" postprepare-envelope "${gradient}" \
        >"${out}/run-${variant}-${gradient}.stdout" \
        2>"${out}/run-${variant}-${gradient}.stderr"
      cmp <(printf 'E3-I2-GUARD-WRAP-PASS\nE3-ZERO-NEGATIVE-RUNTIME-PASS\n') \
        "${out}/run-${variant}-${gradient}.stdout"
      [[ ! -s "${out}/run-${variant}-${gradient}.stderr" ]] ||
        die "${variant}/${gradient} emitted diagnostics"
    done
  done
  cmp "${out}/package-${length}-a/e3_zero_control_package.o" \
      "${out}/package-${length}-b/e3_zero_control_package.o"
  cmp "${out}/package-${length}-a/libeshkol_transformer_e3_zero_control.a" \
      "${out}/package-${length}-b/libeshkol_transformer_e3_zero_control.a"
  for gradient in absent present; do
    cmp "${out}/run-${length}-a-${gradient}.stdout" \
        "${out}/run-${length}-b-${gradient}.stdout"
    cmp "${out}/run-${length}-a-${gradient}.stdout" \
        "${out}/run-${length}-sanitize-${gradient}.stdout"
  done
  E3_I2_POSTPREPARE_GUARD_LENGTH="${length}" \
  E3_I2_POSTPREPARE_MUTANT=omit-abort \
    "${PROJECT_ROOT}/scripts/build-e3-zero-control-test.sh" \
      "${out}/package-${length}-mutant" normal \
      >"${out}/package-${length}-mutant.stdout" \
      2>"${out}/package-${length}-mutant.stderr"
  mutant_object="${out}/package-${length}-mutant/e3_zero_control_package.o"
  mutant_evidence="${mutant_object}.evidence"
  cmp "${PROJECT_ROOT}/tests/e3_private/i2_guard_mutant_package_source_closure.txt" \
      "${mutant_evidence}/source-closure.txt"
  [[ "$(rg -Fxc '@BUILD@/source/i2-guard-overlay/i2_wave2_root.esk' \
      "${mutant_evidence}/source-closure.txt")" == 1 ]] ||
    die "mutant did not load exactly one staged I2 root"
  if rg -Fqx 'native/i2_wave2_root.esk' \
      "${mutant_evidence}/source-closure.txt"; then
    die "mutant also loaded the canonical I2 root"
  fi
  if cmp -s "${out}/package-${length}-a/e3_zero_control_package.o" \
      "${mutant_object}"; then
    die "sole-abort mutant was byte-identical to ordinary package"
  fi
  compile_runtime "${length}-mutant" zero_control_negative 2 "${cxx}"
  ulimit -c 0
  if ESHKOL_ARENA_POISON=0 \
      timeout --foreground --signal=TERM --kill-after=5s 900s \
      "${out}/zero_control_negative-runtime-${length}-mutant" \
      "${out}/oracle-a/all-correct" postprepare-envelope absent \
      >"${out}/run-${length}-mutant.stdout" \
      2>"${out}/run-${length}-mutant.stderr"; then
    die "sole-abort mutant escaped the abort/census assertion"
  else
    mutant_status=$?
  fi
  [[ "${mutant_status}" == 134 ]] ||
    die "sole-abort mutant failed outside the expected assertion"
  rg -Fx 'E3-I2-GUARD-WRAP-FAIL: one prepare, consumed cut, one abort, no commit' \
    "${out}/run-${length}-mutant.stderr" >/dev/null ||
    die "sole-abort mutant missed the real prepared-builder assertion"
done
printf 'E3 I2 postprepare wrapper guard PASS\n'
