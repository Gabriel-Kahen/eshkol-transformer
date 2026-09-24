#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in docker git python3 readlink sha256sum timeout; do
  require_command "${command}"
done

cd "${PROJECT_ROOT}"
candidate_manifest="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
candidate_root="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery}"
[[ "${candidate_root}" = /* && -d "${candidate_root}" ]] || \
  die "frozen TR3 lease runtime root is unavailable: ${candidate_root}"
candidate_root="$(readlink -f -- "${candidate_root}")"

candidate_path() {
  local key=$1 relative resolved
  relative="$(tsv_value "${candidate_manifest}" "${key}")"
  [[ "${relative}" != /* && "${relative}" != *../* && "${relative}" != ../* ]] || \
    die "unsafe ${key} in ${candidate_manifest}: ${relative}"
  resolved="$(readlink -f -- "${candidate_root}/${relative}")"
  [[ "${resolved}" == "${candidate_root}/"* ]] || \
    die "${key} escapes the frozen runtime root"
  printf '%s\n' "${resolved}"
}
verify_hash() {
  local path=$1 key=$2
  [[ -r "${path}" ]] || die "frozen runtime input is unavailable: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == \
     "$(tsv_value "${candidate_manifest}" "${key}")" ]] || \
    die "frozen runtime input hash changed: ${path}"
}

trainer_commit="$(tsv_value "${candidate_manifest}" trainer_source_commit)"
trainer_tree="$(tsv_value "${candidate_manifest}" trainer_source_tree)"
[[ "$(git rev-parse "${trainer_commit}^{tree}")" == "${trainer_tree}" ]] || \
  die "approved TR3 lease source tree changed"
git merge-base --is-ancestor "${trainer_commit}" HEAD || \
  die "checkout does not descend from the approved TR3 lease successor"
git diff --quiet "${trainer_commit}" -- include internal lib native src templates || \
  die "production source differs from the approved TR3 lease successor"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "TR3 lease supported gate requires a clean checkout"
[[ "$(sha256sum native/tr3_lease_core_extension.esk | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" lease_core_sha256)" ]] || \
  die "approved lease core changed"
[[ "$(sha256sum tests/tr3_lease/lease_runtime.esk | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" runtime_fixture_sha256)" ]] || \
  die "approved lease runtime fixture changed"
[[ "$(sha256sum tests/tr3_lease/test_source_contract.py | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" source_contract_sha256)" ]] || \
  die "approved lease source contract changed"
[[ "$(sha256sum tests/tr3_lease/test_restore_successor_contract.py | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" successor_contract_sha256)" ]] || \
  die "approved lease successor contract changed"

candidate_source="$(candidate_path compiler_source_relative_path)"
candidate_runner="$(candidate_path runner_relative_path)"
candidate_archive="$(candidate_path runtime_archive_relative_path)"
accepted_recovery="$(candidate_path accepted_recovery_relative_path)"
compiler_provenance="$(candidate_path compiler_provenance_relative_path)"
final_provenance="$(candidate_path final_provenance_relative_path)"
cmake_cache="$(candidate_path cmake_cache_relative_path)"
[[ -x "${candidate_runner}" ]] || die "frozen TR3 lease runner is not executable"
verify_hash "${candidate_runner}" runner_sha256
verify_hash "${candidate_archive}" runtime_archive_sha256
verify_hash "${accepted_recovery}" accepted_recovery_sha256
verify_hash "${compiler_provenance}" compiler_provenance_sha256
verify_hash "${final_provenance}" final_provenance_sha256
verify_hash "${cmake_cache}" cmake_cache_sha256

compiler_commit="$(tsv_value "${candidate_manifest}" compiler_source_commit)"
compiler_tree="$(tsv_value "${candidate_manifest}" compiler_source_tree)"
[[ "$(git -C "${candidate_source}" rev-parse HEAD)" == "${compiler_commit}" &&
   "$(git -C "${candidate_source}" rev-parse 'HEAD^{tree}')" == "${compiler_tree}" &&
   -z "$(git -C "${candidate_source}" status --porcelain --untracked-files=all)" ]] || \
  die "recovered compiler source is not the exact clean accepted checkout"
[[ "$(tsv_value "${accepted_recovery}" source_commit)" == "${compiler_commit}" &&
   "$(tsv_value "${accepted_recovery}" source_tree)" == "${compiler_tree}" &&
   "$(tsv_value "${accepted_recovery}" accepted_runner_sha256)" == \
     "$(tsv_value "${candidate_manifest}" runner_sha256)" &&
   "$(tsv_value "${accepted_recovery}" runtime_archive_sha256)" == \
     "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)" &&
   "$(tsv_value "${accepted_recovery}" lost_runner_sha256)" == \
     "$(tsv_value "${candidate_manifest}" lost_runner_sha256)" ]] || \
  die "accepted recovery record disagrees with the lease-local manifest"
[[ "$(tsv_value "${candidate_manifest}" runner_role)" == \
     functional_recovered_final81298 &&
   "$(tsv_value "${candidate_manifest}" lost_runner_status)" == missing ]] || \
  die "lease-local runner roles changed"
[[ "$(tsv_value "${compiler_provenance}" eshkol_commit)" == "${compiler_commit}" &&
   "$(tsv_value "${compiler_provenance}" eshkol_binary_sha256)" == \
     "$(tsv_value "${candidate_manifest}" runner_sha256)" &&
   "$(tsv_value "${compiler_provenance}" llvm_version)" == \
     "$(tsv_value "${candidate_manifest}" llvm_version)" ]] || \
  die "compiler provenance disagrees with the lease-local manifest"
[[ "$(tsv_value "${final_provenance}" source_commit)" == "${compiler_commit}" &&
   "$(tsv_value "${final_provenance}" source_tree)" == "${compiler_tree}" &&
   "$(tsv_value "${final_provenance}" eshkol_run_sha256)" == \
     "$(tsv_value "${candidate_manifest}" runner_sha256)" &&
   "$(tsv_value "${final_provenance}" runtime_archive_sha256)" == \
     "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)" &&
   "$(tsv_value "${final_provenance}" source_status)" == clean &&
   "$(tsv_value "${final_provenance}" promotion_testing)" == OFF ]] || \
  die "final runtime provenance disagrees with the lease-local manifest"
for entry in \
  'CMAKE_BUILD_TYPE:STRING=Release' \
  'CMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-21' \
  'CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-21' \
  'ESHKOL_PROMOTION_TESTING:BOOL=OFF'; do
  grep -Fx "${entry}" "${cmake_cache}" >/dev/null || \
    die "recovered runtime profile field changed: ${entry}"
done

candidate_image="$(tsv_value "${candidate_manifest}" container_image)"
expected_image="$(tsv_value "${candidate_manifest}" container_digest)"
[[ "$(docker image inspect "${candidate_image}" --format '{{.Id}}')" == \
   "${expected_image}" ]] || die "TR3 lease candidate container identity changed"
[[ "$(docker run --rm --network none "${expected_image}" llvm-config-21 --version)" == \
   "$(tsv_value "${candidate_manifest}" llvm_version)" ]] || \
  die "TR3 lease candidate LLVM version changed"

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.tr3_lease.test_source_contract \
  tests.tr3_lease.test_restore_successor_contract

explicit_evidence=0
if [[ -n "${TR3_LEASE_EVIDENCE_DIR:-}" ]]; then
  explicit_evidence=1
  temporary_dir="${TR3_LEASE_EVIDENCE_DIR}"
  [[ "${temporary_dir}" = /* ]] || die "TR3_LEASE_EVIDENCE_DIR must be absolute"
  mkdir -p -- "${temporary_dir}"
  [[ -z "$(find "${temporary_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
    die "TR3_LEASE_EVIDENCE_DIR must be empty"
  temporary_dir="$(readlink -f -- "${temporary_dir}")"
else
  temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tr3-lease.XXXXXX")"
fi
cleanup() {
  local status=$?
  if (( status == 0 && explicit_evidence == 0 )); then
    rm -rf -- "${temporary_dir}"
  else
    printf 'TR3 lease evidence retained at %s\n' "${temporary_dir}" >&2
    local log
    shopt -s nullglob
    for log in "${temporary_dir}"/*.stdout "${temporary_dir}"/*.stderr; do
      if [[ -s "${log}" ]]; then
        printf '%s:\n' "${log}" >&2
        sed -n '1,240p' "${log}" >&2
      fi
    done
  fi
  return "${status}"
}
trap cleanup EXIT
mkdir -p "${temporary_dir}/inputs"
cp -- "${candidate_manifest}" \
  scripts/test-tr3-lease.sh \
  tests/tr3_lease/test_source_contract.py \
  tests/tr3_lease/test_restore_successor_contract.py \
  tests/tr3_lease/lease_runtime.esk \
  native/tr3_lease_core_extension.esk \
  "${temporary_dir}/inputs/"
cp -- "${accepted_recovery}" "${compiler_provenance}" \
  "${final_provenance}" "${cmake_cache}" "${temporary_dir}/inputs/"

candidate_runner_relative="$(tsv_value "${candidate_manifest}" runner_relative_path)"
docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate_root}:/candidate:ro" \
  -v "${temporary_dir}:/out" \
  -e "TR3_RUNNER=/candidate/${candidate_runner_relative}" \
  -w /workspace "${candidate_image}" bash -lc '
set -euo pipefail
export ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
export ESHKOL_LIB_DIR=/workspace/lib
timeout --foreground --signal=TERM --kill-after=5s 360s \
  "${TR3_RUNNER}" --strict-types --no-stdlib -O 0 --compile-only \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib tests/tr3_lease/root_smoke.esk \
    -o /out/root_smoke.o > /out/compile.stdout 2> /out/compile.stderr
'
test ! -s "${temporary_dir}/compile.stderr"
[[ -s "${temporary_dir}/root_smoke.o" ]] || \
  die "TR3 lease compile smoke did not produce an object"

PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${temporary_dir}/corpus"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate_root}:/candidate:ro" \
  -v "${temporary_dir}:/out" \
  -e "TR3_RUNNER=/candidate/${candidate_runner_relative}" \
  -w /workspace "${candidate_image}" bash -lc '
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
ar rcsD /out/native/libtr3_lease_runtime.a /out/native/*.o

export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib
export ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
timeout --foreground --signal=TERM --kill-after=5s 900s \
  "${TR3_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib -L /out/native --lib tr3_lease_runtime \
    tests/tr3_lease/lease_runtime.esk -o /out/lease-runtime \
    > /out/runtime-compile.stdout 2> /out/runtime-compile.stderr
for horizon in short long; do
  argument=()
  [[ "${horizon}" == short ]] || argument=(long)
  ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 300s \
    /out/lease-runtime /out/corpus "${argument[@]}" \
    > "/out/${horizon}.stdout" 2> "/out/${horizon}.stderr"
done
'

test ! -s "${temporary_dir}/runtime-compile.stderr"
test ! -s "${temporary_dir}/short.stderr"
test ! -s "${temporary_dir}/long.stderr"
grep -E '^TR3-LEASE-RUNTIME-PASS checks=[0-9]+ horizon=1024 loop_arena_delta=0 trainers=[0-9]+$' \
  "${temporary_dir}/short.stdout" >/dev/null
grep -E '^TR3-LEASE-RUNTIME-PASS checks=[0-9]+ horizon=8192 loop_arena_delta=0 trainers=[0-9]+$' \
  "${temporary_dir}/long.stdout" >/dev/null

{
  printf 'trainer_source_commit\t%s\n' "${trainer_commit}"
  printf 'trainer_source_tree\t%s\n' "${trainer_tree}"
  printf 'gate_checkout_head\t%s\n' "$(git rev-parse HEAD)"
  printf 'gate_checkout_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'compiler_source_commit\t%s\n' "${compiler_commit}"
  printf 'compiler_source_tree\t%s\n' "${compiler_tree}"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runner_sha256)"
  printf 'runner_role\t%s\n' "$(tsv_value "${candidate_manifest}" runner_role)"
  printf 'lost_runner_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" lost_runner_sha256)"
  printf 'lost_runner_status\t%s\n' \
    "$(tsv_value "${candidate_manifest}" lost_runner_status)"
  printf 'runtime_archive_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)"
  printf 'container_digest\t%s\n' "${expected_image}"
  printf 'llvm_version\t%s\n' "$(tsv_value "${candidate_manifest}" llvm_version)"
  printf 'native_archive_sha256\t%s\n' \
    "$(sha256sum "${temporary_dir}/native/libtr3_lease_runtime.a" | awk '{print $1}')"
  printf 'root_smoke_sha256\t%s\n' \
    "$(sha256sum "${temporary_dir}/root_smoke.o" | awk '{print $1}')"
  printf 'lease_runtime_sha256\t%s\n' \
    "$(sha256sum "${temporary_dir}/lease-runtime" | awk '{print $1}')"
} >"${temporary_dir}/provenance.tsv"
(
  cd "${temporary_dir}"
  find . -type f ! -name SHA256SUMS -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)

printf 'TR3 LEASE PASS: recovered exact-81298 strict AOT, authentic successor transitions, and 1024/8192 zero-growth loops\n'
