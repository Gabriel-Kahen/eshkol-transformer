#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in docker python3 sha256sum timeout; do require_command "${command}"; done

cd "${PROJECT_ROOT}"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.tr3_lease.test_source_contract

candidate_manifest="${PROJECT_ROOT}/tests/tr3_p1/runtime_candidate.tsv"
candidate_dir="${TR3_P1_RUNTIME_CANDIDATE_DIR:-/tmp/eshkol-rethrow-final-81298b4a-20260923T200234Z}"
candidate_runner="${candidate_dir}/eshkol-run"
candidate_image="$(tsv_value "${candidate_manifest}" container_image)"
expected_image="$(tsv_value "${candidate_manifest}" container_digest)"

[[ -x "${candidate_runner}" ]] || \
  die "frozen TR3 runtime candidate is unavailable: ${candidate_dir}"
[[ "$(sha256sum "${candidate_runner}" | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" runner_sha256)" ]] || \
  die "TR3 runtime candidate runner hash changed"
[[ "$(docker image inspect "${candidate_image}" --format '{{.Id}}')" == \
   "${expected_image}" ]] || die "TR3 candidate container identity changed"

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tr3-lease.XXXXXX")"
cleanup() {
  local status=$?
  if (( status == 0 )); then
    rm -rf -- "${temporary_dir}"
  else
    printf 'TR3 lease failure evidence retained at %s\n' \
      "${temporary_dir}" >&2
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

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${candidate_dir}:/candidate:ro" \
  -v "${temporary_dir}:/out" \
  -w /workspace "${candidate_image}" bash -lc '
set -euo pipefail
export ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
export ESHKOL_LIB_DIR=/workspace/lib
timeout --foreground --signal=TERM --kill-after=5s 360s \
  /candidate/eshkol-run --strict-types --no-stdlib -O 0 --compile-only \
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
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${candidate_dir}:/candidate:ro" \
  -v "${temporary_dir}:/out" \
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
  /candidate/eshkol-run --strict-types --no-stdlib -O 0 \
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

printf 'TR3 LEASE PASS: structural/privacy checks, strict AOT, authentic tuple acquisition, overlap/idle/gate matrix, and 1024/8192 zero-growth loops\n'
