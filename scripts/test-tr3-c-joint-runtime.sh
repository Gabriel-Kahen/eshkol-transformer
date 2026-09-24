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
  die "frozen TR3 runtime root is unavailable: ${candidate_root}"
candidate_root="$(readlink -f -- "${candidate_root}")"
runner_relative="$(tsv_value "${candidate_manifest}" runner_relative_path)"
runner="${candidate_root}/${runner_relative}"
[[ -x "${runner}" ]] || die "frozen TR3 runner is unavailable"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" runner_sha256)" ]] || \
  die "frozen TR3 runner hash changed"

image="$(tsv_value "${candidate_manifest}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${candidate_manifest}" container_digest)" ]] || \
  die "supported container image digest changed"

evidence="${TR3_JOINT_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-joint-runtime.XXXXXX")}"
mkdir -p "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside the source checkout"

PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate_root}:/candidate:ro" \
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
compile native/p1_identity.c p1_identity.o -DET_P1_TRUSTED_BUILD=1
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
  "${TR3_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib -L /out/native --lib tr3_joint_runtime \
    tests/tr3_joint_restore/runtime_smoke.esk -o /out/joint-runtime \
    > /out/compile.stdout 2> /out/compile.stderr

ESHKOL_ARENA_POISON=1 /usr/bin/time -v -o /out/runtime.time \
  timeout --foreground --signal=TERM --kill-after=5s 300s \
  /out/joint-runtime /out/corpus \
  > /out/runtime.stdout 2> /out/runtime.stderr
'

test ! -s "${evidence}/compile.stderr"
test ! -s "${evidence}/runtime.stderr"
grep -E '^TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry$' \
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
    "$(sha256sum tests/tr3_joint_restore/runtime_smoke.esk | awk '{print $1}')"
  printf 'result\t%s\n' "$(cat "${evidence}/runtime.stdout")"
} >"${evidence}/manifest.tsv"

(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C joint runtime evidence: %s\n' "${evidence}"
printf 'TR3-C joint runtime seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
