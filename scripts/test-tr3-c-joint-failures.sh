#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in docker find git readlink rg sha256sum; do
  require_command "${command}"
done

cd "${PROJECT_ROOT}"
manifest="${PROJECT_ROOT}/tests/tr3_lease/failure_runtime_candidate.tsv"
runtime_root="${TR3_JOINT_FAILURE_RUNTIME_ROOT:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-de0b249-production}"
runtime_source="${TR3_JOINT_FAILURE_RUNTIME_SOURCE:-/home/gabe/.codex/worktrees/wave3-checked-arena-calls/eshkol}"
runtime_root="$(readlink -f -- "${runtime_root}")"
runtime_source="$(readlink -f -- "${runtime_source}")"
runtime_build="${runtime_root}/build"
runner="${runtime_root}/$(tsv_value "${manifest}" runner_relative_path)"
archive="${runtime_root}/$(tsv_value "${manifest}" runtime_archive_relative_path)"
image="$(tsv_value "${manifest}" container_image)"
optimization_level="${TR3_JOINT_OPTIMIZATION_LEVEL:-0}"
case "${optimization_level}" in
  0) compiler_timeout="$(tsv_value "${manifest}" o0_compile_timeout_seconds)" ;;
  2) compiler_timeout="$(tsv_value "${manifest}" o2_compile_timeout_seconds)" ;;
  *) die "joint restore optimization level must be 0 or 2" ;;
esac
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || die "invalid compiler timeout"

[[ -x "${runner}" && -r "${archive}" ]] || \
  die "pinned production-OFF runtime is unavailable"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == \
   "$(tsv_value "${manifest}" runner_sha256)" ]] || die "runner hash changed"
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == \
   "$(tsv_value "${manifest}" runtime_archive_sha256)" ]] || \
  die "runtime archive hash changed"
[[ "$(git -C "${runtime_source}" rev-parse HEAD)" == \
   "$(tsv_value "${manifest}" runtime_source_commit)" ]] || \
  die "runtime source commit changed"
[[ "$(git -C "${runtime_source}" rev-parse 'HEAD^{tree}')" == \
   "$(tsv_value "${manifest}" runtime_source_tree)" ]] || \
  die "runtime source tree changed"
[[ -z "$(git -C "${runtime_source}" status --porcelain --untracked-files=all)" ]] || \
  die "runtime source is dirty"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${manifest}" container_digest)" ]] || die "image digest changed"
[[ "$(tsv_value "${manifest}" runner_role)" == \
   allocation_prefix_production_off ]] || die "runner role changed"

if [[ "${TR3_JOINT_ALLOW_DIRTY:-0}" != 1 ]]; then
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
    die "joint restore checkout must be clean"
fi

evidence="${TR3_JOINT_FAILURE_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-joint-failures.XXXXXX")}"
mkdir -p "${evidence}/inputs"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside the source checkout"
[[ -z "$(find "${evidence}" -mindepth 1 -maxdepth 1 ! -name inputs -print -quit)" ]] || \
  die "evidence directory must be empty"

cp -- scripts/test-tr3-c-joint-failures.sh \
  tests/tr3_joint_restore/allocation_failure_runtime.esk \
  tests/tr3_joint_restore/allocation_failure_shim.cpp \
  tests/tr3_joint_restore/test_source_contract.py \
  tests/tr3_lease/failure_runtime_candidate.tsv \
  "${evidence}/inputs/"

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.tr3_joint_restore.test_source_contract \
  >"${evidence}/source-test.stdout" 2>"${evidence}/source-test.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus" \
  >"${evidence}/fixture.stdout" 2>"${evidence}/fixture.stderr"

set +e
docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${runtime_source}:/runtime-source:ro" \
  -v "${runtime_build}:/runtime-build:ro" \
  -v "${evidence}:/out" \
  -e "TR3_PROBES=${TR3_JOINT_ALLOCATION_PROBES:-}" \
  -e "TR3_OPTIMIZATION_LEVEL=${optimization_level}" \
  -e "TR3_COMPILER_TIMEOUT_SECONDS=${compiler_timeout}" \
  -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/native /out/cache /out/cases
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
       -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
       -fexcess-precision=standard -frounding-math
       -I include -I native -I src)
compile() {
  local source=$1 object=$2
  shift 2
  clang-21 "${flags[@]}" "$@" -c "${source}" -o "/out/native/${object}"
}
compile native/data_io.c data_io.o
compile native/checkpoint_io.c checkpoint_io.o
compile native/kernel_abi.c kernel_abi.o
compile src/eshkol_transformer/m3_i64_integration.c m3_i64_integration.o \
  -DET_I64_TENSOR_TESTING
compile native/t1_i64_shell.c t1_i64_shell.o
compile src/eshkol_transformer/m3t_f32_integration.c m3t_f32_integration.o \
  -DET_F32_TENSOR_TESTING -DET_I2_PRIVATE_OWNED_CLONE_MATCH
compile src/eshkol_transformer/m3_model.c m3_model.o
compile native/d2_native.c d2_native.o \
  -DET_TR3_C_D2_RESTORE -DET_D2_NATIVE_TESTING -DET_I64_TENSOR_TESTING
compile native/n2_primitives_provider.c n2_primitives_provider.o
compile native/n3k_primitives_provider.c n3k_primitives_provider.o
compile native/a2_attention_provider.c a2_attention_provider.o
compile native/indexed_cross_entropy.c indexed_cross_entropy.o
compile native/l3s_masked_objective_provider.c l3s_masked_objective_provider.o
compile native/tr3b_objective_bridge.c tr3b_objective_bridge.o
compile native/o2_optimizer.c o2_optimizer.o \
  -DET_O2_TESTING -DET_TR3_O2_STEP_CLEAR_NATIVE \
  -DET_I2_PRIVATE_OWNED_CLONE_MATCH -DET_TR3_C_O2_RESTORE_NATIVE
compile native/p1_identity.c p1_identity.o -DET_P1_TRUSTED_BUILD=1
compile native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD -DET_F32_TENSOR_TESTING \
  -DET_C2_I2_MODEL_COPY -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE
compile native/o2_wave2_package_bridge.c o2_bridge.o \
  -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -DET_TR3_O2_STEP_CLEAR_BRIDGE
compile native/c2_x1_canonical.c c2_x1_canonical.o
compile native/tr3_c_restore_bindings.c tr3_c_restore_bindings.o \
  -DET_TR3_C_RESTORE_BINDINGS -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE -DET_TR3_C_O2_RESTORE_NATIVE
ar rcsD /out/native/libtr3_joint_failure.a /out/native/*.o

export ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
export ESHKOL_LIB_DIR=/runtime-build XDG_CACHE_HOME=/out/cache
/usr/bin/time -v -o /out/compile.time \
  timeout --foreground --signal=TERM --kill-after=5s \
    "${TR3_COMPILER_TIMEOUT_SECONDS}s" \
  /runtime-build/eshkol-run --strict-types --no-stdlib \
    --optimize "${TR3_OPTIMIZATION_LEVEL}" \
    --emit-object --emit-depfile /out/joint-failure.d \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib tests/tr3_joint_restore/allocation_failure_runtime.esk \
    -o /out/joint-failure.o \
    > /out/compile.stdout 2> /out/compile.stderr

clang++-21 -std=c++17 -O2 -Wall -Wextra -Werror -Wpedantic \
  -Wconversion -Wsign-conversion -Wshadow -fPIC \
  -I /runtime-source/lib/core -I /runtime-source/inc \
  -c tests/tr3_joint_restore/allocation_failure_shim.cpp \
  -o /out/joint-failure-shim.o \
  > /out/shim-compile.stdout 2> /out/shim-compile.stderr

wraps=(
  -Wl,--wrap=arena_allocate
  -Wl,--wrap=arena_allocate_ad_node_with_header
  -Wl,--wrap=arena_allocate_with_header
  -Wl,--wrap=arena_allocate_closure_with_header
  -Wl,--wrap=arena_allocate_string_with_header
  -Wl,--wrap=arena_allocate_vector_with_header
  -Wl,--wrap=arena_allocate_cons_with_header
  -Wl,--wrap=arena_allocate_tensor_with_header
)
clang++-21 -fPIE -fuse-ld=bfd \
  /out/joint-failure.o /out/joint-failure-shim.o \
  /out/native/libtr3_joint_failure.a /runtime-build/libeshkol-runtime.a \
  "${wraps[@]}" -Wl,-Map,/out/joint-failure.map \
  -Wl,-z,stack-size=536870912 -Wl,--export-dynamic \
  -pthread -ldl -lm -lcrypto -lpng -ljpeg -lwebp -lz -lopenblas \
  -o /out/joint-failure \
  > /out/link.stdout 2> /out/link.stderr

nm -g --defined-only /out/joint-failure | awk "{print \$3}" | \
  rg "^__wrap_" | LC_ALL=C sort > /out/linked-wrap-symbols.txt
printf "%s\n" \
  __wrap_arena_allocate \
  __wrap_arena_allocate_ad_node_with_header \
  __wrap_arena_allocate_closure_with_header \
  __wrap_arena_allocate_cons_with_header \
  __wrap_arena_allocate_string_with_header \
  __wrap_arena_allocate_tensor_with_header \
  __wrap_arena_allocate_vector_with_header \
  __wrap_arena_allocate_with_header \
  > /out/expected-wrap-symbols.txt
cmp /out/expected-wrap-symbols.txt /out/linked-wrap-symbols.txt

: > /out/run.stdout
: > /out/run.stderr
printf "class\tordinal\tresult\tstage\texit_status\n" > /out/case-status.tsv
run_case() {
  local class=$1 ordinal=$2 status marker result stage
  set +e
  ESHKOL_ARENA_POISON=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
      /out/joint-failure /out/corpus "${class}" "${ordinal}" \
      > "/out/cases/${class}-${ordinal}.stdout" \
      2> "/out/cases/${class}-${ordinal}.stderr"
  status=$?
  set -e
  cat "/out/cases/${class}-${ordinal}.stdout" >> /out/run.stdout
  cat "/out/cases/${class}-${ordinal}.stderr" >> /out/run.stderr
  if (( status != 0 )); then
    printf "%s\t%s\tPROCESS-FAIL\tunknown\t%s\n" \
      "${class}" "${ordinal}" "${status}" \
      >> /out/case-status.tsv
    return "${status}"
  fi
  marker=$(rg "^TR3-JOINT-ALLOC-CASE class=${class} ordinal=${ordinal} result=" \
    "/out/cases/${class}-${ordinal}.stdout")
  [[ $(printf "%s\n" "${marker}" | wc -l) -eq 1 ]]
  result=$(printf "%s\n" "${marker}" | \
    sed -E "s/^.* result=([^ ]+) stage=.*$/\\1/")
  stage=$(printf "%s\n" "${marker}" | \
    sed -E "s/^.* stage=([0-9]+) checks=.*$/\\1/")
  if [[ "${result}" == HIT ]]; then
    rg "^TR3-JOINT-ALLOC-CUT ordinal=${ordinal} stage=[a-z0-9-]+ allocator=[a-z0-9-]+$" \
      "/out/cases/${class}-${ordinal}.stdout" >/dev/null
  fi
  printf "%s\t%s\t%s\t%s\t0\n" \
    "${class}" "${ordinal}" "${result}" "${stage}" \
    >> /out/case-status.tsv
  case_result=${result}
}

if [[ -n "${TR3_PROBES}" ]]; then
  IFS=, read -r -a probes <<<"${TR3_PROBES}"
  for probe in "${probes[@]}"; do
    class=${probe%%:*}
    ordinal=${probe##*:}
    run_case "${class}" "${ordinal}"
  done
  printf "TR3-JOINT-ALLOCATION-PROBES %s PASS\n" "${TR3_PROBES}" \
    >> /out/run.stdout
else
  total_cases=0
  for class in arena f32 i2 o2; do
    ordinal=0
    while (( ordinal < 8192 )); do
      run_case "${class}" "${ordinal}"
      total_cases=$((total_cases + 1))
      if [[ "${case_result}" == HIT ]]; then
        ordinal=$((ordinal + 1))
      elif [[ "${case_result}" == END ]]; then
        printf "TR3-JOINT-ALLOCATION-CLASS class=%s prefixes=%s cases=%s\n" \
          "${class}" "${ordinal}" "$((ordinal + 1))" >> /out/run.stdout
        break
      else
        exit 86
      fi
    done
    (( ordinal < 8192 )) || exit 87
  done
  printf "TR3-JOINT-ALLOCATION-PASS classes=4 cases=%s\n" \
    "${total_cases}" >> /out/run.stdout
fi
'
container_status=$?
set -e

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'source_status\t%s\n' \
    "$(test -z "$(git status --porcelain --untracked-files=all)" && printf clean || printf dirty)"
  printf 'runtime_commit\t%s\n' "$(tsv_value "${manifest}" runtime_source_commit)"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${manifest}" runner_sha256)"
  printf 'runtime_archive_sha256\t%s\n' \
    "$(tsv_value "${manifest}" runtime_archive_sha256)"
  printf 'runner_role\t%s\n' "$(tsv_value "${manifest}" runner_role)"
  printf 'container_digest\t%s\n' "$(tsv_value "${manifest}" container_digest)"
  printf 'optimization_level\t%s\n' "${optimization_level}"
  printf 'compiler_timeout_seconds\t%s\n' "${compiler_timeout}"
  printf 'allocation_probes\t%s\n' "${TR3_JOINT_ALLOCATION_PROBES:-full-prefix}"
  printf 'container_status\t%s\n' "${container_status}"
  for input in "${evidence}"/inputs/*; do
    printf 'input_sha256\t%s\tinputs/%s\n' \
      "$(sha256sum "${input}" | awk '{print $1}')" "$(basename "${input}")"
  done
} >"${evidence}/manifest.tsv"

(
  cd "${evidence}"
  find . -type f ! -name SHA256SUMS -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)
seal="$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"

if (( container_status != 0 )); then
  printf 'TR3-C joint allocation counterexample evidence: %s\n' "${evidence}" >&2
  printf 'TR3-C joint allocation counterexample seal: %s\n' "${seal}" >&2
  exit "${container_status}"
fi

if [[ "${optimization_level}" == 0 ]]; then
  test ! -s "${evidence}/compile.stderr"
else
  [[ "$(wc -c <"${evidence}/compile.stderr")" == \
     "$(tsv_value "${manifest}" o2_compile_stderr_bytes)" ]]
  [[ "$(sha256sum "${evidence}/compile.stderr" | awk '{print $1}')" == \
     "$(tsv_value "${manifest}" o2_compile_stderr_sha256)" ]]
fi
test ! -s "${evidence}/shim-compile.stderr"
test ! -s "${evidence}/link.stderr"
if rg -n 'fatal signal|TR3 joint allocation shim FAIL|TR3-C JOINT RUNTIME FAIL' \
    "${evidence}/run.stderr"; then
  die "joint allocation witness emitted a fatal diagnostic"
fi
if [[ -z "${TR3_JOINT_ALLOCATION_PROBES:-}" ]]; then
  rg '^TR3-JOINT-ALLOCATION-PASS classes=4 cases=[1-9][0-9]*$' \
    "${evidence}/run.stdout" >/dev/null
fi
printf 'TR3-C joint allocation evidence: %s\n' "${evidence}"
printf 'TR3-C joint allocation seal: %s\n' "${seal}"
