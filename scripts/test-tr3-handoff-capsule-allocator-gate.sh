#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in docker git python3 readlink sha256sum timeout; do
  require_command "${command}"
done

cd "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "source checkout must be clean"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare-tr3-handoff-capsule-allocator-gate.py --verify-only
candidate_manifest="${PROJECT_ROOT}/tests/tr3_handoff_capsule/allocator_prefix_candidate.tsv"
candidate_root="${TR3_CAPSULE_ALLOCATOR_RUNTIME_ROOT:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-de0b249-production}"
runtime_source="${TR3_CAPSULE_ALLOCATOR_RUNTIME_SOURCE:-/home/gabe/.codex/worktrees/wave3-checked-arena-calls/eshkol}"
[[ "${candidate_root}" = /* && -d "${candidate_root}" ]] || \
  die "frozen TR3 runtime root is unavailable: ${candidate_root}"
candidate_root="$(readlink -f -- "${candidate_root}")"
runner_relative="$(tsv_value "${candidate_manifest}" runner_relative_path)"
runner="${candidate_root}/${runner_relative}"
archive="${candidate_root}/$(tsv_value "${candidate_manifest}" runtime_archive_relative_path)"
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)" ]] || die "frozen TR3 runtime archive hash changed"
[[ "$(git -C "${runtime_source}" rev-parse HEAD)" == "$(tsv_value "${candidate_manifest}" runtime_source_commit)" ]] || die "runtime source commit differs"
[[ -z "$(git -C "${runtime_source}" status --porcelain)" ]] || die "runtime source checkout is dirty"
[[ -x "${runner}" ]] || die "frozen TR3 runner is unavailable"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" runner_sha256)" ]] || \
  die "frozen TR3 runner hash changed"

image="$(tsv_value "${candidate_manifest}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${candidate_manifest}" container_digest)" ]] || \
  die "supported container image digest changed"

evidence="${TR3_CAPSULE_ALLOCATOR_EVIDENCE_DIR:-$(mktemp -d "/home/gabe/.codex/evidence/eshkol-transformer/tr3-capsule-allocator.XXXXXX")}"
mkdir -p "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside the source checkout"

PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate_root}:/candidate:ro" \
  -v "${runtime_source}:/runtime-source:ro" \
  -v "${evidence}:/out" \
  -e "TR3_RUNNER=/candidate/${runner_relative}" \
  -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/native /out/cache
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
compile native/p1_identity.c p1_identity.o \
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1
compile native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
  -DET_C2_I2_MODEL_COPY -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE
compile native/o2_wave2_package_bridge.c o2_bridge.o \
  -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -DET_TR3_O2_STEP_CLEAR_BRIDGE
compile native/c2_x1_canonical.c c2_x1_canonical.o
compile native/tr3_c_restore_bindings.c tr3_c_restore_bindings.o \
  -DET_TR3_C_RESTORE_BINDINGS -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE -DET_TR3_C_O2_RESTORE_NATIVE
ar rcsD /out/native/libtr3_joint_runtime.a /out/native/*.o

export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib
export ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
/usr/bin/time -v -o /out/compile.time \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  "${TR3_RUNNER}" --strict-types --no-stdlib -O 0 --emit-object \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib \
    tests/tr3_handoff_capsule/allocator_prefix_runtime.esk \
    -o /out/capsule_allocator.o \
    > /out/compile.stdout 2> /out/compile.stderr
clang++-21 -std=c++17 -O2 -Wall -Wextra -Werror -Wpedantic \
  -Wconversion -Wsign-conversion -Wshadow -fPIC \
  -I /runtime-source/lib/core -I /runtime-source/inc -I include -I native \
  -c tests/tr3_handoff_capsule/capsule_allocator_shim.cpp \
  -o /out/capsule_allocator_shim.o \
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
  -Wl,--wrap=_Znwm
  -Wl,--wrap=_Z28eshkol_region_allocate_quietP5arenamm
  -Wl,--wrap=malloc
  -Wl,--wrap=eshkol_push_exception_handler
  -Wl,--wrap=eshkol_get_raised_value
  -Wl,--wrap=eshkol_region_write_barrier_checked_v1
)
clang++-21 -fPIE -fuse-ld=bfd \
  /out/capsule_allocator.o /out/capsule_allocator_shim.o \
  /out/native/libtr3_joint_runtime.a \
  /candidate/build/libeshkol-runtime.a "${wraps[@]}" \
  -Wl,-Map,/out/capsule_allocator.map \
  -Wl,-z,stack-size=536870912 -Wl,--export-dynamic \
  -pthread -ldl -lm -lcrypto -lpng -ljpeg -lwebp -lz -lopenblas \
  -o /out/capsule_allocator \
  > /out/link.stdout 2> /out/link.stderr
nm -g --defined-only /out/capsule_allocator | awk "{print \$3}" | \
  rg "^__wrap_" | LC_ALL=C sort > /out/linked-wrap-symbols.txt
printf "%s\n" \
  __wrap__Z28eshkol_region_allocate_quietP5arenamm __wrap__Znwm \
  __wrap_arena_allocate __wrap_arena_allocate_ad_node_with_header \
  __wrap_arena_allocate_closure_with_header \
  __wrap_arena_allocate_cons_with_header \
  __wrap_arena_allocate_string_with_header \
  __wrap_arena_allocate_tensor_with_header \
  __wrap_arena_allocate_vector_with_header \
  __wrap_arena_allocate_with_header __wrap_eshkol_get_raised_value \
  __wrap_eshkol_push_exception_handler \
  __wrap_eshkol_region_write_barrier_checked_v1 __wrap_malloc \
  > /out/expected-wrap-symbols.txt
cmp /out/expected-wrap-symbols.txt /out/linked-wrap-symbols.txt
mkdir -p /out/cases
: > /out/runtime.stdout
: > /out/runtime.stderr
printf "class\tordinal\tresult\texit_status\n" > /out/case-status.tsv
run_case() {
  local class=$1 ordinal=$2 case_dir=/out/cases/$1 marker result status
  mkdir -p "${case_dir}"
  set +e
  ESHKOL_ARENA_POISON=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
    /out/capsule_allocator /out/corpus "${class}" "${ordinal}" \
    > "${case_dir}/${ordinal}.stdout" \
    2> "${case_dir}/${ordinal}.stderr"
  status=$?
  set -e
  cat "${case_dir}/${ordinal}.stdout" >> /out/runtime.stdout
  cat "${case_dir}/${ordinal}.stderr" >> /out/runtime.stderr
  if (( status != 0 )); then
    printf "%s\t%s\tPROCESS-FAIL\t%s\n" "${class}" "${ordinal}" "${status}" \
      >> /out/case-status.tsv
    return 85
  fi
  marker=$(rg "^TR3-CAPSULE-ALLOCATOR-CASE class=${class} ordinal=${ordinal} result=(HIT|END) checks=[1-9][0-9]*$" \
    "${case_dir}/${ordinal}.stdout")
  [[ $(printf "%s\n" "${marker}" | wc -l) -eq 1 ]]
  result=${marker#*result=}
  result=${result%% *}
  printf "%s\t%s\t%s\t0\n" "${class}" "${ordinal}" "${result}" \
    >> /out/case-status.tsv
  case_result=${result}
}
for class in object all promotion target; do
  ordinal=0
  while (( ordinal < 2048 )); do
    run_case "${class}" "${ordinal}"
    if [[ "${case_result}" == END ]]; then break; fi
    [[ "${case_result}" == HIT ]]
    ordinal=$((ordinal + 1))
  done
  (( ordinal > 0 && ordinal < 2048 ))
  printf "TR3-CAPSULE-ALLOCATOR-PREFIX class=%s count=%s\n" \
    "${class}" "${ordinal}" >> /out/runtime.stdout
done
run_case handler 0
[[ "${case_result}" == HIT ]]
printf "TR3-CAPSULE-ALLOCATOR-PASS classes=5\n" >> /out/runtime.stdout
'

test ! -s "${evidence}/compile.stderr"
test ! -s "${evidence}/shim-compile.stderr"
test ! -s "${evidence}/link.stderr"
test ! -s "${evidence}/runtime.stderr"
grep -E '^TR3-CAPSULE-ALLOCATOR-PASS classes=5$' \
  "${evidence}/runtime.stdout" >/dev/null

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'source_status\t%s\n' "$(test -z "$(git status --porcelain --untracked-files=all)" && printf clean || printf dirty)"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runner_sha256)"
  printf 'runner_role\t%s\n' "$(tsv_value "${candidate_manifest}" runner_role)"
  printf 'compiler_source_commit\t%s\n' "$(tsv_value "${candidate_manifest}" compiler_source_commit)"
  printf 'container_image\t%s\n' "${image}"
  printf 'container_digest\t%s\n' "$(tsv_value "${candidate_manifest}" container_digest)"
  printf 'runtime_fixture_sha256\t%s\n' \
    "$(sha256sum tests/tr3_handoff_capsule/allocator_prefix_runtime.esk | awk '{print $1}')"
  printf 'result\t%s\n' "$(cat "${evidence}/runtime.stdout")"
} >"${evidence}/manifest.tsv"

(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3 capsule allocator evidence: %s\n' "${evidence}"
printf 'TR3 capsule allocator seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
