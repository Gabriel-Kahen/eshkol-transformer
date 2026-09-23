#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
usage: test-c2-handler-allocation.sh \
  --runtime-source DIR --runtime-build DIR \
  --runner-sha256 SHA256 --archive-sha256 SHA256 \
  [--evidence-dir DIR]

The four required values may instead be supplied with:
  C2_HANDLER_RUNTIME_SOURCE
  C2_HANDLER_RUNTIME_BUILD
  C2_HANDLER_EXPECTED_RUNNER_SHA256
  C2_HANDLER_EXPECTED_ARCHIVE_SHA256

The optional evidence directory may be supplied with C2_HANDLER_EVIDENCE_DIR.
EOF
  exit 2
}

runtime_source="${C2_HANDLER_RUNTIME_SOURCE:-}"
runtime_build="${C2_HANDLER_RUNTIME_BUILD:-}"
expected_runner_sha="${C2_HANDLER_EXPECTED_RUNNER_SHA256:-}"
expected_archive_sha="${C2_HANDLER_EXPECTED_ARCHIVE_SHA256:-}"
evidence_dir="${C2_HANDLER_EVIDENCE_DIR:-}"
while (( $# )); do
  case "$1" in
    --runtime-source)
      (( $# >= 2 )) || usage
      runtime_source=$2
      shift 2
      ;;
    --runtime-build)
      (( $# >= 2 )) || usage
      runtime_build=$2
      shift 2
      ;;
    --runner-sha256)
      (( $# >= 2 )) || usage
      expected_runner_sha=$2
      shift 2
      ;;
    --archive-sha256)
      (( $# >= 2 )) || usage
      expected_archive_sha=$2
      shift 2
      ;;
    --evidence-dir)
      (( $# >= 2 )) || usage
      evidence_dir=$2
      shift 2
      ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

for command in ar awk cmp diff env find git nm python3 readlink rg sed sha256sum \
               sort timeout tr; do
  command -v "${command}" >/dev/null 2>&1 || \
    die "required command not found: ${command}"
done

[[ -n "${runtime_source}" ]] || die "runtime source must be explicit"
[[ -n "${runtime_build}" ]] || die "runtime build must be explicit"
[[ "${runtime_source}" = /* && -d "${runtime_source}" ]] || \
  die "runtime source must be an existing absolute directory: ${runtime_source}"
[[ "${runtime_build}" = /* && -d "${runtime_build}" ]] || \
  die "runtime build must be an existing absolute directory: ${runtime_build}"
runtime_source="$(readlink -f -- "${runtime_source}")"
runtime_build="$(readlink -f -- "${runtime_build}")"
[[ "${expected_runner_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  die "expected runner SHA-256 must be exactly 64 lowercase hexadecimal digits"
[[ "${expected_archive_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  die "expected runtime archive SHA-256 must be exactly 64 lowercase hexadecimal digits"

runner="${runtime_build}/eshkol-run"
runtime_archive="${runtime_build}/libeshkol-runtime.a"
provenance="${runtime_build}/final-provenance.tsv"
checksums="${runtime_build}/SHA256SUMS"
symbol_audit="${runtime_build}/symbol-audit.tsv"
cmake_cache="${runtime_build}/CMakeCache.txt"
[[ -x "${runner}" ]] || die "candidate runner is not executable: ${runner}"
[[ -r "${runtime_archive}" ]] || \
  die "candidate runtime archive is not readable: ${runtime_archive}"
[[ -r "${provenance}" ]] || die "candidate provenance is missing: ${provenance}"
[[ -r "${checksums}" ]] || die "candidate checksum manifest is missing: ${checksums}"
[[ -r "${symbol_audit}" ]] || die "candidate symbol audit is missing: ${symbol_audit}"
[[ -r "${cmake_cache}" ]] || die "candidate CMake cache is missing: ${cmake_cache}"
[[ -r "${runtime_source}/inc/eshkol/eshkol.h" ]] || \
  die "candidate source does not contain inc/eshkol/eshkol.h"

tsv_value() {
  local file=$1 key=$2 count value
  count="$(awk -F '\t' -v key="${key}" '$1 == key { count++ } END { print count + 0 }' "${file}")"
  [[ "${count}" == 1 ]] || die "expected exactly one ${key} entry in ${file}"
  value="$(awk -F '\t' -v key="${key}" '$1 == key { print $2 }' "${file}")"
  [[ -n "${value}" ]] || die "empty ${key} entry in ${file}"
  printf '%s\n' "${value}"
}

actual_runner_sha="$(sha256sum "${runner}" | awk '{ print $1 }')"
actual_archive_sha="$(sha256sum "${runtime_archive}" | awk '{ print $1 }')"
[[ "${actual_runner_sha}" == "${expected_runner_sha}" ]] || \
  die "candidate runner SHA-256 mismatch: expected ${expected_runner_sha}, got ${actual_runner_sha}"
[[ "${actual_archive_sha}" == "${expected_archive_sha}" ]] || \
  die "candidate runtime archive SHA-256 mismatch: expected ${expected_archive_sha}, got ${actual_archive_sha}"
[[ "$(tsv_value "${provenance}" eshkol_run_sha256)" == "${actual_runner_sha}" ]] || \
  die "candidate provenance runner SHA-256 does not match the supplied runner"
[[ "$(tsv_value "${provenance}" runtime_archive_sha256)" == "${actual_archive_sha}" ]] || \
  die "candidate provenance archive SHA-256 does not match the supplied archive"

while read -r expected_sum recorded_path; do
  [[ "${expected_sum}" =~ ^[0-9a-f]{64}$ && -n "${recorded_path}" ]] || \
    die "malformed candidate SHA256SUMS entry"
  artifact="${runtime_build}/${recorded_path##*/}"
  [[ -f "${artifact}" ]] || die "candidate checksum artifact is missing: ${artifact}"
  [[ "$(sha256sum "${artifact}" | awk '{ print $1 }')" == "${expected_sum}" ]] || \
    die "candidate checksum mismatch: ${artifact}"
done <"${checksums}"

cmp <(printf '%s\n' \
  'ESHKOL_PROMOTION_TESTING:BOOL=OFF' \
  $'present-all\teshkol_region_write_barrier_checked_v1' \
  $'present-all\teshkol_runtime_emergency_raise_v1' \
  $'present-all\teshkol_runtime_emergency_rethrow_if_v1' \
  $'present-all\teshkol_root_arena_v1' \
  $'present-all\teshkol_runtime_reserve_exception_handlers_v1' \
  $'absent-all\teshkol_region_write_barrier_range' \
  $'absent-all\teshkol_promotion_test_' \
  $'closed\truntime-emergency-rethrow-modifier-aot') "${symbol_audit}" || \
  die "candidate production symbol audit differs from the reviewed contract"

candidate_commit="$(tsv_value "${provenance}" source_commit)"
candidate_tree="$(tsv_value "${provenance}" source_tree)"
[[ "${candidate_commit}" =~ ^[0-9a-f]{40}$ ]] || \
  die "candidate provenance commit is not a full Git object ID"
[[ "${candidate_tree}" =~ ^[0-9a-f]{40}$ ]] || \
  die "candidate provenance tree is not a full Git object ID"
[[ "$(tsv_value "${provenance}" source_status)" == clean ]] || \
  die "candidate provenance did not record a clean source checkout"
[[ "$(tsv_value "${provenance}" build_type)" == Release ]] || \
  die "candidate provenance did not record a Release build"
[[ "$(tsv_value "${provenance}" promotion_testing)" == OFF ]] || \
  die "candidate provenance did not record a production runtime"
[[ "$(awk -F= '$1 == "CMAKE_BUILD_TYPE:STRING" { print $2 }' "${cmake_cache}")" == Release ]] || \
  die "candidate CMake cache did not record a Release build"
[[ "$(awk -F= '$1 == "ESHKOL_PROMOTION_TESTING:BOOL" { print $2 }' "${cmake_cache}")" == OFF ]] || \
  die "candidate CMake cache enabled promotion testing"

cc="$(tsv_value "${provenance}" cc)"
cxx="$(tsv_value "${provenance}" cxx)"
[[ "${cc}" = /* && -x "${cc}" && "${cxx}" = /* && -x "${cxx}" ]] || \
  die "candidate provenance must name executable absolute C and C++ compilers"
cc_version="$(${cc} --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
cxx_version="$(${cxx} --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')"
[[ "${cc_version}" == 21.1.8 && "${cxx_version}" == 21.1.8 && \
    "$(tsv_value "${provenance}" llvm_version)" == 21.1.8 ]] || \
  die "allocation witness requires the reviewed Clang/LLVM 21.1.8 toolchain"

git_commit=unavailable
git_tree=unavailable
if git -C "${runtime_source}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_commit="$(git -C "${runtime_source}" rev-parse HEAD)"
  git_tree="$(git -C "${runtime_source}" rev-parse 'HEAD^{tree}')"
  [[ "${git_commit}" == "${candidate_commit}" ]] || \
    die "candidate source HEAD ${git_commit} does not match build provenance ${candidate_commit}"
  [[ "${git_tree}" == "${candidate_tree}" ]] || \
    die "candidate source tree ${git_tree} does not match build provenance ${candidate_tree}"
  [[ -z "$(git -C "${runtime_source}" status --porcelain --untracked-files=all)" ]] || \
    die "candidate runtime source checkout is not clean"
fi

automatic_evidence=0
if [[ -z "${evidence_dir}" ]]; then
  evidence_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-c2-handler-allocation.XXXXXX")"
  automatic_evidence=1
else
  [[ "${evidence_dir}" = /* ]] || die "evidence directory must be absolute"
  mkdir -p -- "${evidence_dir}"
  [[ -z "$(find "${evidence_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
    die "evidence directory must be empty: ${evidence_dir}"
  evidence_dir="$(readlink -f -- "${evidence_dir}")"
fi
cleanup() {
  local status=$?
  if (( status != 0 )); then
    {
      printf 'candidate_commit\t%s\n' "${candidate_commit}"
      printf 'candidate_tree\t%s\n' "${candidate_tree}"
      printf 'runner_sha256\t%s\n' "${actual_runner_sha}"
      printf 'runtime_archive_sha256\t%s\n' "${actual_archive_sha}"
      while IFS= read -r -d '' path; do
        printf '%s\t%s\n' "$(sha256sum "${path}" | awk '{ print $1 }')" \
          "${path#"${evidence_dir}/"}"
      done < <(find "${evidence_dir}" -type f \
        ! -name failure-sha256.tsv -print0 | LC_ALL=C sort -z)
    } >"${evidence_dir}/failure-sha256.tsv"
    find "${evidence_dir}" -type f \( -name '*.stdout' -o -name '*.stderr' \) \
      -print -exec sed -n '1,260p' {} \; >&2 || true
  fi
  if (( automatic_evidence )); then
    rm -rf -- "${evidence_dir}"
  else
    printf 'C2 handler allocation evidence: %s\n' "${evidence_dir}" >&2
  fi
  return "${status}"
}
trap cleanup EXIT

fixtures="${evidence_dir}/fixtures-a"
runtime_objects="${evidence_dir}/runtime"
mkdir -p "${fixtures}" "${runtime_objects}" "${evidence_dir}/cache"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_load_fixtures.py" "${fixtures}"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${PROJECT_ROOT}/tests/c2/prepare_checkpoint_load_fixtures.py" \
  "${evidence_dir}/fixtures-b"
diff -ru "${fixtures}" "${evidence_dir}/fixtures-b"

cflags=(
  -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion
  -Wsign-conversion -Wshadow -ffp-contract=off
  -fexcess-precision=standard -frounding-math -fPIC -fvisibility=hidden
  -fno-common -fstack-protector-all -I "${PROJECT_ROOT}/include"
  -I "${PROJECT_ROOT}/native" -DET_F32_TENSOR_TESTING -DET_O2_TESTING
)
"${cc}" "${cflags[@]}" -DET_I2_NATIVE_HELPERS_ONLY \
  -c "${PROJECT_ROOT}/native/i2_wave2_package_bridge.c" \
  -o "${runtime_objects}/i2.o"
"${cc}" "${cflags[@]}" -DET_O2_NATIVE_HELPERS_ONLY \
  -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -c "${PROJECT_ROOT}/native/o2_wave2_package_bridge.c" \
  -o "${runtime_objects}/o2.o"
"${cc}" "${cflags[@]}" -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1 \
  -c "${PROJECT_ROOT}/native/p1_identity.c" -o "${runtime_objects}/p1.o"
for source in data_io kernel_abi t1_i64_shell f32_tensor i64_tensor d2_native \
              o2_optimizer c2_x1_canonical c2_checkpoint_format; do
  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \
    -o "${runtime_objects}/${source}.o"
done
"${cc}" "${cflags[@]}" -DET_C2_CARRIER_FACTORIES \
  -c "${PROJECT_ROOT}/native/k2_capabilities.c" -o "${runtime_objects}/k2.o"
"${cc}" "${cflags[@]}" -DET_CHECKPOINT_IO_TESTING \
  -c "${PROJECT_ROOT}/native/checkpoint_io.c" \
  -o "${runtime_objects}/checkpoint_io.o"
"${cc}" "${cflags[@]}" -DET_C2_CHECKPOINT_READER_TESTING \
  -c "${PROJECT_ROOT}/native/c2_checkpoint_reader.c" \
  -o "${runtime_objects}/reader.o"
"${cc}" "${cflags[@]}" -DET_C2_CHECKPOINT_CORE_TESTING \
  -c "${PROJECT_ROOT}/native/c2_checkpoint_core.c" \
  -o "${runtime_objects}/core.o"
"${cc}" "${cflags[@]}" -DET_C2_CHECKPOINT_LOAD_TESTING \
  -c "${PROJECT_ROOT}/native/c2_checkpoint_load_bridge.c" \
  -o "${runtime_objects}/load.o"
"${cc}" "${cflags[@]}" \
  -c "${PROJECT_ROOT}/tests/c2/c2_checkpoint_save_test_bridge.c" \
  -o "${runtime_objects}/test.o"
ar rcsD "${runtime_objects}/libeshkol_transformer_c2_handler.a" \
  "${runtime_objects}"/*.o

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -Wconversion \
  -Wsign-conversion -Wshadow -fPIC -I "${runtime_source}/inc" \
  -c "${PROJECT_ROOT}/tests/c2/handler_allocation_shim.cpp" \
  -o "${evidence_dir}/handler_allocation_shim.o"

compile_stderr="${evidence_dir}/compile.stderr"
compile_stdout="${evidence_dir}/compile.stdout"
env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
  XDG_CACHE_HOME="${evidence_dir}/cache" ESHKOL_LIB_DIR="${runtime_build}" \
  ESHKOL_CXX_COMPILER="${cxx}" \
  timeout --foreground --signal=TERM --kill-after=5s 900s "${runner}" \
    --strict-types --optimize 0 --no-stdlib --emit-object \
    --emit-depfile "${evidence_dir}/handler_allocation.d" \
    -I "${PROJECT_ROOT}/internal/p1/lib" \
    -I "${PROJECT_ROOT}/internal/c1/lib" \
    -I "${PROJECT_ROOT}/internal/t2/lib" \
    -I "${PROJECT_ROOT}/internal/t1/lib" \
    -I "${PROJECT_ROOT}/internal/d2/lib" \
    -I "${PROJECT_ROOT}/src" -I "${PROJECT_ROOT}/lib" \
    -I "${PROJECT_ROOT}/native" \
    "${PROJECT_ROOT}/tests/c2/handler_allocation.esk" \
    -o "${evidence_dir}/handler_allocation.o" \
    >"${compile_stdout}" 2>"${compile_stderr}"
test ! -s "${compile_stderr}" || die "strict AOT compilation wrote stderr"

sed -e 's/^[^:]*://' -e 's/\\//g' "${evidence_dir}/handler_allocation.d" | \
  tr -s '[:space:]' '\n' | rg "^${PROJECT_ROOT}/" | \
  sed "s#^${PROJECT_ROOT}/##" | \
  rg -v '^tests/c2/handler_allocation\.esk$' \
  >"${evidence_dir}/source-closure.txt"
cmp "${PROJECT_ROOT}/native/c2_checkpoint_load_source_closure.txt" \
  "${evidence_dir}/source-closure.txt" || \
  die "handler allocation Eshkol source dependency closure drifted"

wrap_flags=(
  -Wl,--wrap=malloc
  -Wl,--wrap=eshkol_push_exception_handler
  -Wl,--wrap=eshkol_get_raised_value
  -Wl,--wrap=eshkol_runtime_reserve_exception_handlers_v1
  -Wl,--wrap=et_p1_private_state_bind_v1
  -Wl,--wrap=et_p1_private_state_release_begin_v1
  -Wl,--wrap=et_i2_private_owned_release_v1
  -Wl,--wrap=et_o2_private_optimizer_state_release_v1
)
nm -u "${evidence_dir}/handler_allocation_shim.o" | awk '{ print $2 }' | \
  rg '^__real_' | LC_ALL=C sort >"${evidence_dir}/shim-real-symbols.txt"
printf '%s\n' \
  __real_eshkol_get_raised_value \
  __real_eshkol_push_exception_handler \
  __real_eshkol_runtime_reserve_exception_handlers_v1 \
  __real_et_i2_private_owned_release_v1 \
  __real_et_o2_private_optimizer_state_release_v1 \
  __real_et_p1_private_state_bind_v1 \
  __real_et_p1_private_state_release_begin_v1 \
  __real_malloc >"${evidence_dir}/expected-real-symbols.txt"
cmp "${evidence_dir}/expected-real-symbols.txt" \
  "${evidence_dir}/shim-real-symbols.txt" || \
  die "shim does not declare the exact reviewed GNU __real_ symbol set"
"${cxx}" -fPIE -fuse-ld=bfd "${evidence_dir}/handler_allocation.o" \
  "${evidence_dir}/handler_allocation_shim.o" \
  -L "${runtime_objects}" -leshkol_transformer_c2_handler \
  "${runtime_archive}" \
  "${wrap_flags[@]}" -Wl,-Map,"${evidence_dir}/handler_allocation.map" \
  -Wl,-z,stack-size=536870912 -Wl,--export-dynamic \
  -pthread -ldl -lm -lcrypto -lpng -ljpeg -lwebp -lz \
  -o "${evidence_dir}/handler_allocation"

nm -g --defined-only "${evidence_dir}/handler_allocation" | \
  awk '{ print $3 }' | rg '^__wrap_' | LC_ALL=C sort \
  >"${evidence_dir}/linked-wrap-symbols.txt"
sed 's/^__real_/__wrap_/' "${evidence_dir}/expected-real-symbols.txt" \
  >"${evidence_dir}/expected-wrap-symbols.txt"
cmp "${evidence_dir}/expected-wrap-symbols.txt" \
  "${evidence_dir}/linked-wrap-symbols.txt" || \
  die "linked witness does not define the exact reviewed GNU __wrap_ symbol set"
rg -F 'libeshkol-runtime.a(runtime_exceptions_hosted.cpp.o)' \
  "${evidence_dir}/handler_allocation.map" >/dev/null || \
  die "link map does not bind exception handlers to the supplied runtime archive"
for object in p1.o i2.o o2.o; do
  rg -F "libeshkol_transformer_c2_handler.a(${object})" \
    "${evidence_dir}/handler_allocation.map" >/dev/null || \
    die "link map does not bind ${object} from the compiled package archive"
done

expected_stdout="${PROJECT_ROOT}/tests/c2/expected/handler_allocation.stdout"
[[ "$(sha256sum "${expected_stdout}" | awk '{ print $1 }')" == \
    351a1cf1c2625dc7edc275950a8c91542ccd7c5bbddd1e2313acda19db634784 ]] || \
  die "checked-in exact witness stdout changed without review"

run_stdout="${evidence_dir}/run.stdout"
run_stderr="${evidence_dir}/run.stderr"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 360s \
  "${evidence_dir}/handler_allocation" "${fixtures}/valid.c2" \
  >"${run_stdout}" 2>"${run_stderr}"
test ! -s "${run_stderr}" || die "handler allocation witness wrote stderr"
cmp "${expected_stdout}" "${run_stdout}" || \
  die "handler allocation witness stdout did not match the exact ten-case record"

repeat_stdout="${evidence_dir}/run-repeat.stdout"
repeat_stderr="${evidence_dir}/run-repeat.stderr"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 360s \
  "${evidence_dir}/handler_allocation" "${fixtures}/valid.c2" \
  >"${repeat_stdout}" 2>"${repeat_stderr}"
test ! -s "${repeat_stderr}" || die "repeated handler allocation witness wrote stderr"
cmp "${run_stdout}" "${repeat_stdout}" || \
  die "repeated handler allocation witness stdout changed"

"${cxx}" -std=c++17 -Wall -Wextra -Werror -Wpedantic -Wconversion \
  -Wsign-conversion -Wshadow -fPIC \
  -DET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10=1 \
  -I "${runtime_source}/inc" \
  -c "${PROJECT_ROOT}/tests/c2/handler_allocation_shim.cpp" \
  -o "${evidence_dir}/handler_allocation_diagnostic_shim.o"
"${cxx}" -fPIE -fuse-ld=bfd "${evidence_dir}/handler_allocation.o" \
  "${evidence_dir}/handler_allocation_diagnostic_shim.o" \
  -L "${runtime_objects}" -leshkol_transformer_c2_handler \
  "${runtime_archive}" "${wrap_flags[@]}" \
  -Wl,-Map,"${evidence_dir}/handler_allocation_diagnostic.map" \
  -Wl,-z,stack-size=536870912 -Wl,--export-dynamic \
  -pthread -ldl -lm -lcrypto -lpng -ljpeg -lwebp -lz \
  -o "${evidence_dir}/handler_allocation_diagnostic"
diagnostic_stdout="${evidence_dir}/diagnostic.stdout"
diagnostic_stderr="${evidence_dir}/diagnostic.stderr"
set +e
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 360s \
  "${evidence_dir}/handler_allocation_diagnostic" "${fixtures}/valid.c2" \
  >"${diagnostic_stdout}" 2>"${diagnostic_stderr}"
diagnostic_status=$?
set -e
[[ "${diagnostic_status}" == 97 ]] || \
  die "reserve-10 diagnostic exited ${diagnostic_status}, expected 97"
sed -n '1,9p' "${expected_stdout}" >"${evidence_dir}/diagnostic-expected.stdout"
cmp "${evidence_dir}/diagnostic-expected.stdout" "${diagnostic_stdout}" || \
  die "reserve-10 diagnostic did not complete the first nine exact cases"
printf '%s\n' \
  'C2-HANDLER-DIAGNOSTIC case=3 mode=3 reserve=11/10 push=19 active=10 bind=1 provider=0/1/0 acquire=1/0/0 EXPECTED' \
  >"${evidence_dir}/diagnostic-expected.stderr"
cmp "${evidence_dir}/diagnostic-expected.stderr" "${diagnostic_stderr}" || \
  die "reserve-10 diagnostic did not hit the exact LOAD dual-fault allocation"

manifest="${evidence_dir}/sha256.tsv"
{
  printf 'candidate_commit\t%s\n' "${candidate_commit}"
  printf 'candidate_tree\t%s\n' "${candidate_tree}"
  printf 'runner_sha256\t%s\n' "${actual_runner_sha}"
  printf 'runtime_archive_sha256\t%s\n' "${actual_archive_sha}"
  for entry in \
    "witness_source:${PROJECT_ROOT}/tests/c2/handler_allocation.esk" \
    "witness_shim:${PROJECT_ROOT}/tests/c2/handler_allocation_shim.cpp" \
    "source_closure:${evidence_dir}/source-closure.txt" \
    "native_archive:${runtime_objects}/libeshkol_transformer_c2_handler.a" \
    "aot_object:${evidence_dir}/handler_allocation.o" \
    "executable:${evidence_dir}/handler_allocation" \
    "compile_stdout:${compile_stdout}" \
    "compile_stderr:${compile_stderr}" \
    "run_stdout:${run_stdout}" \
    "run_stderr:${run_stderr}" \
    "run_repeat_stdout:${repeat_stdout}" \
    "run_repeat_stderr:${repeat_stderr}" \
    "diagnostic_shim:${evidence_dir}/handler_allocation_diagnostic_shim.o" \
    "diagnostic_executable:${evidence_dir}/handler_allocation_diagnostic" \
    "diagnostic_stdout:${diagnostic_stdout}" \
    "diagnostic_stderr:${diagnostic_stderr}"; do
    label=${entry%%:*}
    path=${entry#*:}
    printf '%s_sha256\t%s\n' "${label}" "$(sha256sum "${path}" | awk '{ print $1 }')"
  done
} >"${manifest}"
cat "${manifest}"
printf '%s\n' \
  'C2 HANDLER ALLOCATION PASS: explicit repaired runtime, Clang 21 strict AOT O0, exact source closure, GNU allocation/cleanup wrappers, 10 exact cases twice, reserve-10 LOAD discrimination'
