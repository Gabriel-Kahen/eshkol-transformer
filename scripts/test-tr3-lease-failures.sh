#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
tsv_value() {
  local file=$1 key=$2 value count
  count=$(awk -F '\t' -v key="${key}" '$1 == key { count++ } END { print count + 0 }' "${file}")
  [[ "${count}" == 1 ]] || die "expected one ${key} entry in ${file}; found ${count}"
  value=$(awk -F '\t' -v key="${key}" '$1 == key { print $2 }' "${file}")
  [[ -n "${value}" ]] || die "empty ${key} entry in ${file}"
  printf '%s\n' "${value}"
}
usage() {
  cat >&2 <<'EOF'
usage: test-tr3-lease-failures.sh \
  --runtime-source DIR --runtime-build DIR \
  --runner-sha256 SHA256 --archive-sha256 SHA256 \
  [--runtime-commit COMMIT --runtime-tree TREE] \
  [--production-base COMMIT] \
  [--allocation-class object|all] [--optimize 0|2] \
  [--allocation-probes 1094,1095] \
  [--diagnostic-only] [--evidence-dir DIR]
EOF
  exit 2
}

failure_manifest="${PROJECT_ROOT}/tests/tr3_lease/failure_runtime_candidate.tsv"
runtime_root="${TR3_LEASE_FAILURE_RUNTIME_ROOT:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-de0b249-production}"
runtime_source="${TR3_LEASE_FAILURE_RUNTIME_SOURCE:-/home/gabe/.codex/worktrees/wave3-checked-arena-calls/eshkol}"
runtime_build="${runtime_root}/build"
runner_sha="$(tsv_value "${failure_manifest}" runner_sha256)"
archive_sha="$(tsv_value "${failure_manifest}" runtime_archive_sha256)"
evidence_dir=""
runtime_commit="$(tsv_value "${failure_manifest}" runtime_source_commit)"
runtime_tree="$(tsv_value "${failure_manifest}" runtime_source_tree)"
allocation_class=all
optimize=0
production_base=""
diagnostic_only=0
allocation_probes=""
while (( $# )); do
  case "$1" in
    --runtime-source) runtime_source=$2; shift 2 ;;
    --runtime-build) runtime_build=$2; shift 2 ;;
    --runner-sha256) runner_sha=$2; shift 2 ;;
    --archive-sha256) archive_sha=$2; shift 2 ;;
    --runtime-commit) runtime_commit=$2; shift 2 ;;
    --runtime-tree) runtime_tree=$2; shift 2 ;;
    --allocation-class) allocation_class=$2; shift 2 ;;
    --optimize) optimize=$2; shift 2 ;;
    --production-base) production_base=$2; shift 2 ;;
    --allocation-probes) allocation_probes=$2; shift 2 ;;
    --diagnostic-only) diagnostic_only=1; shift ;;
    --evidence-dir) evidence_dir=$2; shift 2 ;;
    *) usage ;;
  esac
done

for tool in awk cmp docker find git python3 readlink rg sed sha256sum wc; do
  command -v "${tool}" >/dev/null || die "required command not found: ${tool}"
done
[[ "${runtime_source}" = /* && -d "${runtime_source}" ]] || usage
[[ "${runtime_build}" = /* && -d "${runtime_build}" ]] || usage
[[ "${runner_sha}" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "${runtime_commit}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${runtime_tree}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${allocation_class}" == object || "${allocation_class}" == all ]] || usage
[[ "${optimize}" == 0 || "${optimize}" == 2 ]] || usage
[[ -z "${allocation_probes}" || "${allocation_probes}" == 1094,1095 ]] || usage
[[ -z "${allocation_probes}" || "${allocation_class}" == all ]] || usage
(( diagnostic_only == 0 )) || [[ -z "${allocation_probes}" ]] || usage
if [[ "${allocation_class}" == all ]]; then
  [[ "${runtime_commit}" == "$(tsv_value "${failure_manifest}" runtime_source_commit)" &&
     "${runtime_tree}" == "$(tsv_value "${failure_manifest}" runtime_source_tree)" ]] || \
    die "full allocator matrix requires the reviewed production-OFF runtime"
  historical_trainer_commit="$(tsv_value "${failure_manifest}" trainer_source_commit)"
  historical_trainer_tree="$(tsv_value "${failure_manifest}" trainer_source_tree)"
  [[ "$(git -C "${PROJECT_ROOT}" rev-parse \
       "${historical_trainer_commit}^{tree}")" == "${historical_trainer_tree}" ]] || \
    die "historical failed-matrix trainer identity changed"
  expected_production_base="$(tsv_value "${failure_manifest}" approved_production_commit)"
  expected_production_tree="$(tsv_value "${failure_manifest}" approved_production_tree)"
  expected_witness_commit="${expected_production_base}"
  expected_witness_tree="${expected_production_tree}"
  [[ "${runner_sha}" == "$(tsv_value "${failure_manifest}" runner_sha256)" &&
     "${archive_sha}" == "$(tsv_value "${failure_manifest}" runtime_archive_sha256)" ]] || \
    die "full matrix requires the lease-local reviewed runtime artifacts"
  [[ "$(tsv_value "${failure_manifest}" runner_role)" == \
     allocation_prefix_production_off ]] || \
    die "failure runner role changed"
else
  die "the successor recovery gate supports only the reviewed full allocator matrix"
fi
production_base=${production_base:-${expected_production_base}}
[[ "${production_base}" == "${expected_production_base}" ]] || \
  die "production base is not the reviewed source for this matrix"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse "${production_base}^{tree}")" == \
   "${expected_production_tree}" ]] || \
  die "reviewed production tree identity changed"
expected_runner_self_sha256=bde887e0897c55b92e76be21e6b3a6c5f10a3a025032d35de2e2b035da9c7c27
runner_self_sha256="$(sed \
  's/^expected_runner_self_sha256=.*/expected_runner_self_sha256=__SELF__/' \
  "${PROJECT_ROOT}/scripts/test-tr3-lease-failures.sh" | \
  sha256sum | awk '{print $1}')"
[[ "${runner_self_sha256}" == "${expected_runner_self_sha256}" ]] || \
  die "canonical lease failure runner changed"
expected_source_test_sha256="$(tsv_value "${failure_manifest}" source_contract_sha256)"
expected_successor_test_sha256="$(tsv_value "${failure_manifest}" successor_contract_sha256)"
[[ "$(sha256sum \
  "${PROJECT_ROOT}/tests/tr3_lease/test_source_contract.py" | awk '{print $1}')" == \
   "${expected_source_test_sha256}" ]] || \
  die "reviewed lease source-contract test changed"
[[ "$(sha256sum \
  "${PROJECT_ROOT}/tests/tr3_lease/test_restore_successor_contract.py" | awk '{print $1}')" == \
   "${expected_successor_test_sha256}" ]] || \
  die "reviewed lease successor-contract test changed"
[[ "$(sha256sum \
  "${PROJECT_ROOT}/native/tr3_lease_core_extension.esk" | awk '{print $1}')" == \
   "$(tsv_value "${failure_manifest}" lease_core_sha256)" ]] || \
  die "reviewed lease core changed"
[[ "$(sha256sum \
  "${PROJECT_ROOT}/internal/d2/lib/d2_semantic_core.esk" | awk '{print $1}')" == \
   "$(tsv_value "${failure_manifest}" d2_semantic_core_sha256)" ]] || \
  die "reviewed D2 semantic core changed"
[[ "$(sha256sum \
  "${PROJECT_ROOT}/tests/d2/semantic_core.esk" | awk '{print $1}')" == \
   "$(tsv_value "${failure_manifest}" d2_semantic_fixture_sha256)" ]] || \
  die "reviewed D2 semantic fixture changed"
[[ "$(sha256sum \
  "${PROJECT_ROOT}/tests/d2/test_scope.py" | awk '{print $1}')" == \
   "$(tsv_value "${failure_manifest}" d2_scope_test_sha256)" ]] || \
  die "reviewed D2 scope test changed"
[[ "$(sha256sum \
  "${PROJECT_ROOT}/tests/tr3_lease_failure/lease_failure_runtime.esk" | awk '{print $1}')" == \
   "$(tsv_value "${failure_manifest}" failure_fixture_sha256)" ]] || \
  die "reviewed successor failure fixture changed"
runtime_source="$(readlink -f -- "${runtime_source}")"
runtime_build="$(readlink -f -- "${runtime_build}")"
runtime_root="$(readlink -f -- "$(dirname -- "${runtime_build}")")"
manifest_path() {
  local key=$1 relative resolved
  relative="$(tsv_value "${failure_manifest}" "${key}")"
  [[ "${relative}" != /* && "${relative}" != *../* && "${relative}" != ../* ]] || \
    die "unsafe ${key} in ${failure_manifest}: ${relative}"
  resolved="$(readlink -f -- "${runtime_root}/${relative}")"
  [[ "${resolved}" == "${runtime_root}/"* ]] || \
    die "${key} escapes the reviewed failure runtime root"
  printf '%s\n' "${resolved}"
}
production_paths=(
  include internal lib native src
)
git -C "${PROJECT_ROOT}" diff --quiet \
  "${production_base}" -- \
  "${production_paths[@]}" || die "frozen production lease sources changed"
if [[ "${allocation_class}" == all ]]; then
  [[ "$(git -C "${PROJECT_ROOT}" rev-parse \
     "${expected_witness_commit}^{tree}")" == \
     "${expected_witness_tree}" ]] || \
    die "reviewed lease witness tree identity changed"
  git -C "${PROJECT_ROOT}" merge-base --is-ancestor \
    "${expected_witness_commit}" HEAD || \
    die "lease checkout does not descend from the reviewed witness"
  git -C "${PROJECT_ROOT}" diff --quiet "${expected_witness_commit}" -- \
    include internal lib native src templates tests/d2 \
    tests/tr3_lease_failure/lease_failure_shim.cpp || \
    die "reviewed lease production source or native witness changed"
  while IFS= read -r changed; do
    case "${changed}" in
      docs/*|\
      scripts/test-tr3-lease.sh|\
      scripts/test-tr3-lease-failures.sh|\
      tests/tr3_lease/test_source_contract.py|\
      tests/tr3_lease/lease_runtime.esk|\
      tests/tr3_lease/runtime_candidate.tsv|\
      tests/tr3_lease/failure_runtime_candidate.tsv|\
      tests/tr3_lease_failure/lease_failure_runtime.esk|\
      tests/tr3_lease_failure/README.md) ;;
      *) die "unreviewed checkout change: ${changed}" ;;
    esac
  done < <(git -C "${PROJECT_ROOT}" diff --name-only \
           "${expected_witness_commit}" --)
fi
[[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]] || \
  die "successor checkout must be clean"
[[ -z "$(git -C "${PROJECT_ROOT}" ls-files --others --exclude-standard -- \
  "${production_paths[@]}")" ]] || die "untracked production lease source found"
runner="$(manifest_path runner_relative_path)"
archive="$(manifest_path runtime_archive_relative_path)"
root_provenance="$(manifest_path runtime_provenance_relative_path)"
build_provenance="$(manifest_path build_provenance_relative_path)"
cmake_cache="$(manifest_path cmake_cache_relative_path)"
runtime_manifest="$(manifest_path artifact_manifest_relative_path)"
[[ "$(dirname -- "${runner}")" == "${runtime_build}" &&
   "$(dirname -- "${archive}")" == "${runtime_build}" ]] || \
  die "runtime build does not match the reviewed manifest layout"
[[ -x "${runner}" && -r "${archive}" ]] || die "runtime artifacts unavailable"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == "${runner_sha}" ]] || \
  die "runner hash mismatch"
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]] || \
  die "runtime archive hash mismatch"
[[ "$(sha256sum "${root_provenance}" | awk '{print $1}')" == \
   "$(tsv_value "${failure_manifest}" runtime_provenance_sha256)" ]] || \
  die "reviewed runtime root provenance changed"
[[ "$(tsv_value "${root_provenance}" source_commit)" == "${runtime_commit}" &&
   "$(tsv_value "${root_provenance}" source_tree)" == "${runtime_tree}" &&
   "$(tsv_value "${root_provenance}" runner_sha256)" == "${runner_sha}" &&
   "$(tsv_value "${root_provenance}" runtime_archive_sha256)" == "${archive_sha}" &&
   "$(tsv_value "${root_provenance}" container_image_id)" == \
     "$(tsv_value "${failure_manifest}" container_digest)" &&
   "$(tsv_value "${root_provenance}" promotion_testing)" == OFF ]] || \
  die "runtime root provenance disagrees with the failure manifest"
if [[ -r "${build_provenance}" ]]; then
  build_commit="$(awk -F '\t' '$1 == "eshkol_commit" {print $2}' \
    "${build_provenance}")"
  build_runner_sha="$(awk -F '\t' '$1 == "eshkol_binary_sha256" {print $2}' \
    "${build_provenance}")"
elif [[ -r "${runtime_build}/final-provenance.tsv" ]]; then
  build_commit="$(awk -F '\t' '$1 == "source_commit" {print $2}' \
    "${runtime_build}/final-provenance.tsv")"
  build_runner_sha="$(awk -F '\t' '$1 == "eshkol_run_sha256" {print $2}' \
    "${runtime_build}/final-provenance.tsv")"
else
  die "runtime build provenance is unavailable"
fi
[[ "${build_commit}" == "${runtime_commit}" ]] || \
  die "runtime build provenance commit mismatch"
[[ "${build_runner_sha}" == "${runner_sha}" ]] || \
  die "runtime build provenance runner hash mismatch"
[[ "$(git -C "${runtime_source}" rev-parse HEAD)" == \
   "${runtime_commit}" ]] || \
  die "runtime source commit mismatch"
[[ "$(git -C "${runtime_source}" rev-parse 'HEAD^{tree}')" == \
   "${runtime_tree}" ]] || \
  die "runtime source tree mismatch"
[[ -z "$(git -C "${runtime_source}" status --porcelain --untracked-files=all)" ]] || \
  die "runtime source is not clean"
grep -Fx 'ESHKOL_PROMOTION_TESTING:BOOL=OFF' \
  "${cmake_cache}" >/dev/null || \
  die "runtime build enabled promotion testing"
if [[ "${allocation_class}" == all ]]; then
  [[ "$(sha256sum "${cmake_cache}" | awk '{print $1}')" == \
     "$(tsv_value "${failure_manifest}" cmake_cache_sha256)" ]] || \
    die "reviewed runtime build profile changed"
  [[ "$(sha256sum \
     "${build_provenance}" | awk '{print $1}')" == \
     "$(tsv_value "${failure_manifest}" build_provenance_sha256)" ]] || \
    die "runtime build provenance changed"
  [[ "$(sha256sum "${runtime_manifest}" | awk '{print $1}')" == \
     "$(tsv_value "${failure_manifest}" artifact_manifest_sha256)" ]] || \
    die "reviewed runtime checksum manifest changed"
  (cd "${runtime_root}" && sha256sum -c "${runtime_manifest}" >/dev/null) || \
    die "reviewed runtime checksum manifest does not verify"
  for entry in \
    'CMAKE_BUILD_TYPE:STRING=Release' \
    'CMAKE_C_COMPILER:STRING=/usr/bin/clang-21' \
    'CMAKE_CXX_COMPILER:STRING=/usr/bin/clang++-21' \
    'ESHKOL_REQUIRED_LLVM_MAJOR:UNINITIALIZED=21' \
    'ESHKOL_BUILD_TESTS:BOOL=OFF' \
    'ESHKOL_PROMOTION_TESTING:BOOL=OFF'; do
    grep -Fx "${entry}" "${cmake_cache}" >/dev/null || \
      die "reviewed runtime profile field changed: ${entry}"
  done
fi
container_id="$(docker image inspect "$(tsv_value "${failure_manifest}" container_image)" \
  --format '{{.Id}}')"
[[ "${container_id}" == "$(tsv_value "${failure_manifest}" container_digest)" ]] || \
  die "supported container identity changed"
if [[ "${allocation_class}" == all ]]; then
  llvm_version="$(docker run --rm --network none "${container_id}" \
    llvm-config-21 --version)"
  [[ "${llvm_version}" == "$(tsv_value "${failure_manifest}" llvm_version)" ]] || \
    die "supported LLVM version changed"
  "${PROJECT_ROOT}/scripts/generate-p1-roots.sh" --check || \
    die "generated P1 root differs from its reviewed template"
  python3 "${PROJECT_ROOT}/scripts/check-tr3-p1-fixed.py" >/dev/null || \
    die "P1 fixed-set structure check failed"
fi

automatic=0
if [[ -z "${evidence_dir}" ]]; then
  evidence_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tr3-lease-failures.XXXXXX")"
  automatic=1
else
  [[ "${evidence_dir}" = /* ]] || die "evidence directory must be absolute"
  mkdir -p -- "${evidence_dir}"
  [[ -z "$(find "${evidence_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
    die "evidence directory must be empty"
  evidence_dir="$(readlink -f -- "${evidence_dir}")"
fi

mkdir -p "${evidence_dir}/inputs"
cp -- \
  "${PROJECT_ROOT}/scripts/test-tr3-lease-failures.sh" \
  "${PROJECT_ROOT}/docs/ROADMAP.md" \
  "${PROJECT_ROOT}/tests/tr3_lease/test_source_contract.py" \
  "${PROJECT_ROOT}/tests/tr3_lease/test_restore_successor_contract.py" \
  "${PROJECT_ROOT}/tests/tr3_lease/lease_runtime.esk" \
  "${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv" \
  "${PROJECT_ROOT}/tests/tr3_lease/failure_runtime_candidate.tsv" \
  "${PROJECT_ROOT}/tests/tr3_lease_failure/README.md" \
  "${PROJECT_ROOT}/tests/tr3_lease_failure/lease_failure_runtime.esk" \
  "${PROJECT_ROOT}/tests/tr3_lease_failure/lease_failure_shim.cpp" \
  "${PROJECT_ROOT}/tests/d2/semantic_core.esk" \
  "${PROJECT_ROOT}/tests/d2/public_errors_runtime.esk" \
  "${PROJECT_ROOT}/tests/d2/test_scope.py" \
  "${evidence_dir}/inputs/"
if [[ "${allocation_class}" == all ]]; then
  mkdir -p "${evidence_dir}/inputs/source"
  cp -- \
    "${PROJECT_ROOT}/native/x1_config_private.esk" \
    "${PROJECT_ROOT}/native/tr3_lease_core_extension.esk" \
    "${PROJECT_ROOT}/internal/d2/lib/d2_semantic_core.esk" \
    "${PROJECT_ROOT}/internal/d2/lib/d2_dataset.esk" \
    "${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk" \
    "${PROJECT_ROOT}/templates/p1/module_roots.esk.tmpl" \
    "${evidence_dir}/inputs/source/"
  cp -- "${cmake_cache}" \
    "${evidence_dir}/inputs/runtime-CMakeCache.txt"
  cp -- "${root_provenance}" \
    "${evidence_dir}/inputs/runtime-provenance.tsv"
  cp -- "${build_provenance}" \
    "${evidence_dir}/inputs/runtime-build-provenance.tsv"
  cp -- "${runtime_manifest}" \
    "${evidence_dir}/inputs/runtime-artifact-sha256.txt"
fi

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.tr3_lease.test_source_contract \
  tests.tr3_lease.test_restore_successor_contract \
  >"${evidence_dir}/source-test.stdout" \
  2>"${evidence_dir}/source-test.stderr"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence_dir}/corpus" \
  >"${evidence_dir}/fixture.stdout" 2>"${evidence_dir}/fixture.stderr"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${runtime_source}:/runtime-source:ro" \
  -v "${runtime_build}:/runtime-build:ro" \
  -v "${evidence_dir}:/out" \
  -e "TR3_ALLOCATION_CLASS=${allocation_class}" \
  -e "TR3_ALLOCATION_PROBES=${allocation_probes}" \
  -e "TR3_OPTIMIZE=${optimize}" \
  -e "TR3_DIAGNOSTIC_ONLY=${diagnostic_only}" \
  -w /workspace "${container_id}" bash -lc '
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
  -DET_F32_TENSOR_TESTING
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
  -DET_O2_TESTING -DET_TR3_O2_STEP_CLEAR_NATIVE
compile native/p1_identity.c p1_identity.o -DET_P1_TRUSTED_BUILD=1
compile native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD
compile native/o2_wave2_package_bridge.c o2_bridge.o \
  -DET_O2_NATIVE_HELPERS_ONLY -DET_TR3_O2_STEP_CLEAR_BRIDGE
ar rcsD /out/native/libtr3_lease_failure_runtime.a /out/native/*.o

export ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
export ESHKOL_LIB_DIR=/runtime-build XDG_CACHE_HOME=/out/cache
timeout --foreground --signal=TERM --kill-after=5s 900s \
  /runtime-build/eshkol-run --strict-types --no-stdlib \
    --optimize "${TR3_OPTIMIZE}" \
    --emit-object --emit-depfile /out/lease_failure.d \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib tests/tr3_lease_failure/lease_failure_runtime.esk \
    -o /out/lease_failure.o \
    > /out/compile.stdout 2> /out/compile.stderr

clang++-21 -std=c++17 -O2 -Wall -Wextra -Werror -Wpedantic \
  -Wconversion -Wsign-conversion -Wshadow -fPIC \
  -I /runtime-source/lib/core -I /runtime-source/inc \
  -I include -I native \
  -c tests/tr3_lease_failure/lease_failure_shim.cpp \
  -o /out/lease_failure_shim.o \
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
  /out/lease_failure.o /out/lease_failure_shim.o \
  /out/native/libtr3_lease_failure_runtime.a \
  /runtime-build/libeshkol-runtime.a "${wraps[@]}" \
  -Wl,-Map,/out/lease_failure.map \
  -Wl,-z,stack-size=536870912 -Wl,--export-dynamic \
  -pthread -ldl -lm -lcrypto -lpng -ljpeg -lwebp -lz -lopenblas \
  -o /out/lease_failure \
  > /out/link.stdout 2> /out/link.stderr

nm -g --defined-only /out/lease_failure | awk "{print \$3}" | \
  rg "^__wrap_" | LC_ALL=C sort > /out/linked-wrap-symbols.txt
printf "%s\n" \
  __wrap__Z28eshkol_region_allocate_quietP5arenamm \
  __wrap__Znwm \
  __wrap_arena_allocate \
  __wrap_arena_allocate_ad_node_with_header \
  __wrap_arena_allocate_closure_with_header \
  __wrap_arena_allocate_cons_with_header \
  __wrap_arena_allocate_string_with_header \
  __wrap_arena_allocate_tensor_with_header \
  __wrap_arena_allocate_vector_with_header \
  __wrap_arena_allocate_with_header \
  __wrap_eshkol_get_raised_value \
  __wrap_eshkol_push_exception_handler \
  __wrap_eshkol_region_write_barrier_checked_v1 \
  __wrap_malloc > /out/expected-wrap-symbols.txt
cmp /out/expected-wrap-symbols.txt /out/linked-wrap-symbols.txt
if [[ "${TR3_DIAGNOSTIC_ONLY}" == 1 ]]; then
  exit 0
fi
mkdir -p /out/cases
: > /out/run.stdout
: > /out/run.stderr
printf "class\tordinal\tresult\texit_status\n" > /out/case-status.tsv
run_case() {
  local class=$1 ordinal=$2 case_dir=/out/cases/$1
  local status marker result
  mkdir -p "${case_dir}"
  set +e
  ESHKOL_ARENA_POISON=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    timeout --foreground --signal=TERM --kill-after=5s 120s \
      /out/lease_failure /out/corpus "${class}" "${ordinal}" \
      > "${case_dir}/${ordinal}.stdout" \
      2> "${case_dir}/${ordinal}.stderr"
  status=$?
  set -e
  cat "${case_dir}/${ordinal}.stdout" >> /out/run.stdout
  cat "${case_dir}/${ordinal}.stderr" >> /out/run.stderr
  if (( status != 0 )); then
    printf "%s\t%s\tPROCESS-FAIL\t%s\n" \
      "${class}" "${ordinal}" "${status}" >> /out/case-status.tsv
    return "${status}"
  fi
  marker=$(rg "^TR3-LEASE-CASE class=${class} ordinal=${ordinal} result=" \
    "${case_dir}/${ordinal}.stdout")
  [[ $(printf "%s\n" "${marker}" | wc -l) -eq 1 ]]
  result=$(printf "%s\n" "${marker}" | \
    sed -E "s/^.* result=([^ ]+) checks=.*$/\\1/")
  printf "%s\t%s\t%s\t0\n" \
    "${class}" "${ordinal}" "${result}" >> /out/case-status.tsv
  case_result=${result}
}
sweep() {
  local class=$1 ordinal=0
  while (( ordinal < 2048 )); do
    run_case "${class}" "${ordinal}"
    if [[ "${case_result}" == HIT ]]; then
      ordinal=$((ordinal + 1))
    elif [[ "${case_result}" == END ]]; then
      printf -v "${class}_count" "%d" "${ordinal}"
      return 0
    else
      printf "invalid matrix result: %s %s %s\n" \
        "${class}" "${ordinal}" "${case_result}" >&2
      return 85
    fi
  done
  printf "matrix prefix limit exceeded: %s\n" "${class}" >&2
  return 86
}
if [[ -n "${TR3_ALLOCATION_PROBES}" ]]; then
  IFS=, read -r -a probe_ordinals <<<"${TR3_ALLOCATION_PROBES}"
  for ordinal in "${probe_ordinals[@]}"; do
    run_case "${TR3_ALLOCATION_CLASS}" "${ordinal}"
    [[ "${case_result}" == HIT ]]
  done
  printf "TR3-LEASE-FAILURE-PROBES class=%s ordinals=%s PASS\n" \
    "${TR3_ALLOCATION_CLASS}" "${TR3_ALLOCATION_PROBES}" >> /out/run.stdout
else
  sweep "${TR3_ALLOCATION_CLASS}"
  sweep promotion
  sweep target
  run_case handler 0
  [[ "${case_result}" == HIT ]]
  run_case poststage 0
  [[ "${case_result}" == PASS ]]
  total_checks=$(awk -F "checks=" "/^TR3-LEASE-CASE / {sum += \$2} END {print sum}" \
    /out/run.stdout)
  allocation_count_var="${TR3_ALLOCATION_CLASS}_count"
  allocation_count="${!allocation_count_var}"
  total_cases=$((allocation_count + promotion_count + target_count + 5))
  printf "TR3-LEASE-FAILURE-CLASSES %s=%s promotion=%s target=%s " \
    "${TR3_ALLOCATION_CLASS}" "${allocation_count}" \
    "${promotion_count}" "${target_count}" \
    >> /out/run.stdout
  printf "handler=1 PASS\n" >> /out/run.stdout
  printf "TR3-LEASE-FAILURE-PASS checks=%s cases=%s\n" \
    "${total_checks}" "${total_cases}" >> /out/run.stdout
fi
'

cmp -- "${PROJECT_ROOT}/scripts/test-tr3-lease-failures.sh" \
  "${evidence_dir}/inputs/test-tr3-lease-failures.sh"
cmp -- "${PROJECT_ROOT}/docs/ROADMAP.md" \
  "${evidence_dir}/inputs/ROADMAP.md"
cmp -- "${PROJECT_ROOT}/tests/tr3_lease/test_source_contract.py" \
  "${evidence_dir}/inputs/test_source_contract.py"
cmp -- "${PROJECT_ROOT}/tests/tr3_lease/test_restore_successor_contract.py" \
  "${evidence_dir}/inputs/test_restore_successor_contract.py"
cmp -- "${PROJECT_ROOT}/tests/tr3_lease/lease_runtime.esk" \
  "${evidence_dir}/inputs/lease_runtime.esk"
cmp -- "${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv" \
  "${evidence_dir}/inputs/runtime_candidate.tsv"
cmp -- "${failure_manifest}" \
  "${evidence_dir}/inputs/failure_runtime_candidate.tsv"
cmp -- "${PROJECT_ROOT}/tests/tr3_lease_failure/README.md" \
  "${evidence_dir}/inputs/README.md"
cmp -- "${PROJECT_ROOT}/tests/tr3_lease_failure/lease_failure_runtime.esk" \
  "${evidence_dir}/inputs/lease_failure_runtime.esk"
cmp -- "${PROJECT_ROOT}/tests/tr3_lease_failure/lease_failure_shim.cpp" \
  "${evidence_dir}/inputs/lease_failure_shim.cpp"
cmp -- "${PROJECT_ROOT}/tests/d2/semantic_core.esk" \
  "${evidence_dir}/inputs/semantic_core.esk"
cmp -- "${PROJECT_ROOT}/tests/d2/public_errors_runtime.esk" \
  "${evidence_dir}/inputs/public_errors_runtime.esk"
cmp -- "${PROJECT_ROOT}/tests/d2/test_scope.py" \
  "${evidence_dir}/inputs/test_scope.py"
if [[ "${allocation_class}" == all ]]; then
  for source in \
    native/x1_config_private.esk \
    native/tr3_lease_core_extension.esk \
    internal/d2/lib/d2_semantic_core.esk \
    internal/d2/lib/d2_dataset.esk \
    internal/p1/lib/transformer/module.esk \
    templates/p1/module_roots.esk.tmpl; do
    cmp -- "${PROJECT_ROOT}/${source}" \
      "${evidence_dir}/inputs/source/${source##*/}"
  done
  cmp -- "${cmake_cache}" \
    "${evidence_dir}/inputs/runtime-CMakeCache.txt"
  cmp -- "${root_provenance}" \
    "${evidence_dir}/inputs/runtime-provenance.tsv"
  cmp -- "${build_provenance}" \
    "${evidence_dir}/inputs/runtime-build-provenance.tsv"
  cmp -- "${runtime_manifest}" \
    "${evidence_dir}/inputs/runtime-artifact-sha256.txt"
fi

grep -Fx 'OK' "${evidence_dir}/source-test.stderr" >/dev/null
if [[ "${optimize}" == 2 ]]; then
  expected_o2_bytes="$(tsv_value "${failure_manifest}" o2_compile_stderr_bytes)"
  expected_o2_sha="$(tsv_value "${failure_manifest}" o2_compile_stderr_sha256)"
  if (( diagnostic_only == 0 )); then
    [[ "${expected_o2_bytes}" =~ ^[0-9]+$ && "${expected_o2_sha}" =~ ^[0-9a-f]{64}$ ]] || \
      die "O2 diagnostics require reviewed byte-count and SHA pins"
    [[ "$(wc -c < "${evidence_dir}/compile.stderr")" == "${expected_o2_bytes}" ]] || \
      die "O2 compiler diagnostics differ in length from the reviewed warnings"
    [[ "$(sha256sum "${evidence_dir}/compile.stderr" | awk '{print $1}')" == \
       "${expected_o2_sha}" ]] || \
      die "O2 compiler diagnostics differ from the reviewed warnings"
  fi
else
  test ! -s "${evidence_dir}/compile.stderr"
fi
test ! -s "${evidence_dir}/shim-compile.stderr"
test ! -s "${evidence_dir}/link.stderr"
if (( diagnostic_only == 1 )); then
  {
    printf 'frozen_production_base_commit\t%s\n' "${production_base}"
    printf 'frozen_production_base_tree\t%s\n' "${expected_production_tree}"
    printf 'runtime_source_commit\t%s\n' "${runtime_commit}"
    printf 'runtime_source_tree\t%s\n' "${runtime_tree}"
    printf 'runner_sha256\t%s\n' "${runner_sha}"
    printf 'runner_role\t%s\n' "$(tsv_value "${failure_manifest}" runner_role)"
    printf 'runtime_archive_sha256\t%s\n' "${archive_sha}"
    printf 'runtime_provenance_sha256\t%s\n' \
      "$(sha256sum "${root_provenance}" | awk '{print $1}')"
    printf 'failure_runtime_candidate_sha256\t%s\n' \
      "$(sha256sum "${failure_manifest}" | awk '{print $1}')"
    printf 'optimization_level\t%s\n' "${optimize}"
    printf 'diagnostic_only\t1\n'
    printf 'container_image_id\t%s\n' "${container_id}"
    while IFS= read -r input; do
      printf 'input_sha256\t%s\t%s\n' \
        "$(sha256sum "${input}" | awk '{print $1}')" \
        "${input#"${evidence_dir}/"}"
    done < <(find "${evidence_dir}/inputs" -type f | LC_ALL=C sort)
  } >"${evidence_dir}/provenance.tsv"
  (
    cd "${evidence_dir}"
    find . -type f ! -name SHA256SUMS -print0 | \
      LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS >/dev/null
  )
  printf 'TR3 lease O%s diagnostic evidence PASS: %s\n' "${optimize}" "${evidence_dir}"
  exit 0
fi
if rg -n 'fatal signal|TR3 lease failure shim FAIL|TR3 LEASE FAILURE FAIL' \
    "${evidence_dir}/run.stderr"; then
  die "runtime witness emitted a fatal diagnostic"
fi
if [[ -n "${allocation_probes}" ]]; then
  for ordinal in 1094 1095; do
    grep -E "^TR3-LEASE-CASE class=all ordinal=${ordinal} result=HIT checks=[1-9][0-9]*$" \
      "${evidence_dir}/run.stdout" >/dev/null
  done
  grep -Fx 'TR3-LEASE-FAILURE-PROBES class=all ordinals=1094,1095 PASS' \
    "${evidence_dir}/run.stdout" >/dev/null
else
  grep -E "^TR3-LEASE-FAILURE-CLASSES ${allocation_class}=[1-9][0-9]* promotion=[1-9][0-9]* target=[1-9][0-9]* handler=1 PASS$" \
    "${evidence_dir}/run.stdout" >/dev/null
  if [[ "${allocation_class}" == object ]]; then
    grep -E '^TR3-LEASE-UNSAFE-ALLOCATOR-CENSUS raw=[1-9][0-9]* header=[0-9]+ closure=[1-9][0-9]* string=[1-9][0-9]* tensor=0 ad-node=0 BLOCKED$' \
      "${evidence_dir}/run.stdout" >/dev/null
  else
    grep -E '^TR3-LEASE-ALL-ALLOCATOR-CENSUS vector=[1-9][0-9]* cons=[1-9][0-9]* raw=[1-9][0-9]* header=[0-9]+ closure=[1-9][0-9]* string=[1-9][0-9]* tensor=[0-9]+ ad-node=[0-9]+ PASS$' \
      "${evidence_dir}/run.stdout" >/dev/null
  fi
  grep -E '^TR3-LEASE-FAILURE-PASS checks=[1-9][0-9]* cases=[1-9][0-9]*$' \
    "${evidence_dir}/run.stdout" >/dev/null
fi

{
  printf 'frozen_production_base_commit\t%s\n' "${production_base}"
  printf 'frozen_production_base_tree\t%s\n' "${expected_production_tree}"
  printf 'reviewed_witness_commit\t%s\n' "${expected_witness_commit:-historical}"
  printf 'reviewed_witness_tree\t%s\n' "${expected_witness_tree:-historical}"
  printf 'runner_self_sha256\t%s\n' "${runner_self_sha256}"
  printf 'successor_checkout_head\t%s\n' \
    "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  printf 'successor_checkout_tree\t%s\n' \
    "$(git -C "${PROJECT_ROOT}" rev-parse 'HEAD^{tree}')"
  printf 'successor_checkout_status\t%s\n' \
    "$(git -C "${PROJECT_ROOT}" status --short | tr '\n' ';')"
  printf 'runtime_source_commit\t%s\n' "$(git -C "${runtime_source}" rev-parse HEAD)"
  printf 'runtime_source_tree\t%s\n' "${runtime_tree}"
  printf 'runner_sha256\t%s\n' "${runner_sha}"
  printf 'runner_role\t%s\n' "$(tsv_value "${failure_manifest}" runner_role)"
  printf 'runtime_archive_sha256\t%s\n' "${archive_sha}"
  printf 'failure_runtime_candidate_sha256\t%s\n' \
    "$(sha256sum "${failure_manifest}" | awk '{print $1}')"
  printf 'diagnostic_only\t%s\n' "${diagnostic_only}"
  printf 'runtime_manifest_sha256\t%s\n' \
    "$(if [[ "${allocation_class}" == all ]]; then \
         sha256sum "${runtime_manifest}" | awk '{print $1}'; \
       else printf 'historical'; fi)"
  printf 'runtime_build_profile_sha256\t%s\n' \
    "$(sha256sum "${cmake_cache}" | awk '{print $1}')"
  printf 'llvm_version\t%s\n' "${llvm_version:-historical}"
  printf 'allocation_class\t%s\n' "${allocation_class}"
  printf 'allocation_probes\t%s\n' "${allocation_probes:-full-matrix}"
  printf 'optimization_level\t%s\n' "${optimize}"
  printf 'container_image_id\t%s\n' "${container_id}"
  while IFS= read -r input; do
    printf 'input_sha256\t%s\t%s\n' \
      "$(sha256sum "${input}" | awk '{print $1}')" \
      "${input#"${evidence_dir}/"}"
  done < <(find "${evidence_dir}/inputs" -type f | LC_ALL=C sort)
} >"${evidence_dir}/provenance.tsv"
(
  cd "${evidence_dir}"
  find . -type f ! -name SHA256SUMS -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)

printf 'TR3 lease failure evidence PASS: %s\n' "${evidence_dir}"
if (( automatic )); then
  printf 'automatic evidence retained for review\n'
fi
