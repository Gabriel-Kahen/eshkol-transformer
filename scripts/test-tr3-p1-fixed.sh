#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar awk cmp docker grep sha256sum; do require_command "${command}"; done
candidate_manifest="${PROJECT_ROOT}/tests/tr3_p1/runtime_candidate.tsv"
candidate_dir="${TR3_P1_RUNTIME_CANDIDATE_DIR:-/tmp/eshkol-rethrow-final-81298b4a-20260923T200234Z}"
candidate_runner="${candidate_dir}/eshkol-run"
candidate_archive="${candidate_dir}/libeshkol-runtime.a"
candidate_image="$(tsv_value "${candidate_manifest}" container_image)"
expected_image="$(tsv_value "${candidate_manifest}" container_digest)"

[[ -x "${candidate_runner}" && -f "${candidate_archive}" ]] || \
  die "frozen TR3 P1 runtime candidate is unavailable: ${candidate_dir}"
[[ "$(sha256sum "${candidate_runner}" | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" runner_sha256)" ]] || \
  die "TR3 P1 runtime candidate runner hash changed"
[[ "$(sha256sum "${candidate_archive}" | awk '{print $1}')" == \
   "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)" ]] || \
  die "TR3 P1 runtime candidate archive hash changed"
[[ "$(docker image inspect "${candidate_image}" --format '{{.Id}}')" == \
   "${expected_image}" ]] || die "TR3 P1 candidate container identity changed"
grep -Fx "source_commit$(printf '\t')$(tsv_value "${candidate_manifest}" source_commit)" \
  "${candidate_dir}/final-provenance.tsv" >/dev/null || \
  die "TR3 P1 candidate source commit changed"
grep -Fx "source_tree$(printf '\t')$(tsv_value "${candidate_manifest}" source_tree)" \
  "${candidate_dir}/final-provenance.tsv" >/dev/null || \
  die "TR3 P1 candidate source tree changed"

python3 "${PROJECT_ROOT}/scripts/check-tr3-p1-fixed.py"
"${PROJECT_ROOT}/scripts/generate-p1-roots.sh" --check

evidence_root="${TR3_P1_EVIDENCE_ROOT:-$(project_build_dir)}"
mkdir -p "${evidence_root}"
evidence_dir="$(mktemp -d "${evidence_root}/tr3-p1-fixed-candidate.XXXXXX")"
relative_evidence="${evidence_dir#"${PROJECT_ROOT}/"}"
[[ "${relative_evidence}" != "${evidence_dir}" ]] || \
  die "TR3 P1 evidence directory must be within the project mount"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${candidate_dir}:/candidate:ro" \
  -w /workspace "${candidate_image}" bash -lc '
set -euo pipefail
out="$1"
mkdir -p "${out}/native"
flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
       -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
       -fexcess-precision=standard -frounding-math
       -I include -I native -I src)
compile() { local source=$1 object=$2; shift 2; clang-21 "${flags[@]}" "$@" -c "${source}" -o "${out}/native/${object}"; }
compile native/a2_attention_provider.c a2_attention_provider.o
compile native/checkpoint_io.c checkpoint_io.o
compile native/data_io.c data_io.o
compile native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD
compile native/kernel_abi.c kernel_abi.o
compile src/eshkol_transformer/m3_i64_integration.c m3_i64_integration.o
compile src/eshkol_transformer/m3_model.c m3_model.o
compile src/eshkol_transformer/m3t_f32_integration.c m3t_f32_integration.o \
  -DET_F32_TENSOR_TESTING
compile native/n2_primitives_provider.c n2_primitives_provider.o
compile native/n3k_primitives_provider.c n3k_primitives_provider.o
compile native/p1_identity.c p1_identity.o \
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1
compile native/t1_i64_shell.c t1_i64_shell.o
ar rcsD "${out}/native/libtr3_p1_fixed_test_runtime.a" "${out}"/native/*.o
ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER=/usr/bin/clang++-21 \
  /candidate/eshkol-run --strict-types --no-stdlib -O 0 \
  -I native -I internal/p1/lib -I internal/c1/lib -I internal/t1/lib -I src \
  -L "${out}/native" --lib tr3_p1_fixed_test_runtime \
  tests/tr3_p1/fixed_set_smoke.esk -o "${out}/fixed-set"
for horizon in short long; do
  argument=()
  [[ "${horizon}" == short ]] || argument=(long)
  ESHKOL_ARENA_POISON=1 ESHKOL_ARENA_REPORT=1 \
    "${out}/fixed-set" "${argument[@]}" \
    >"${out}/${horizon}.stdout" 2>"${out}/${horizon}.stderr"
done
' bash "${relative_evidence}"

grep -Fx 'TR3-P1-FIXED-PASS checks=55 horizon=1024 capture_arena_delta=0 recheck_arena_delta=0 loop_arena_delta=0 p1_live_delta=0 p1_tombstone_delta=0 f32_fail_after=0' \
  "${evidence_dir}/short.stdout" >/dev/null
grep -Fx 'TR3-P1-FIXED-PASS checks=55 horizon=8192 capture_arena_delta=0 recheck_arena_delta=0 loop_arena_delta=0 p1_live_delta=0 p1_tombstone_delta=0 f32_fail_after=0' \
  "${evidence_dir}/long.stdout" >/dev/null
short_bytes="$(awk -F= '/global_total_allocated_bytes=/{print $2}' "${evidence_dir}/short.stderr" | tail -1)"
long_bytes="$(awk -F= '/global_total_allocated_bytes=/{print $2}' "${evidence_dir}/long.stderr" | tail -1)"
[[ "${short_bytes}" =~ ^[0-9]+$ && "${short_bytes}" == "${long_bytes}" ]] || \
  die "TR3 P1 retained arena differs: 1024=${short_bytes:-missing} 8192=${long_bytes:-missing}"

{
  printf 'runtime_commit\t%s\n' "$(tsv_value "${candidate_manifest}" source_commit)"
  printf 'runtime_tree\t%s\n' "$(tsv_value "${candidate_manifest}" source_tree)"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runner_sha256)"
  printf 'short_arena_bytes\t%s\n' "${short_bytes}"
  printf 'long_arena_bytes\t%s\n' "${long_bytes}"
} >"${evidence_dir}/RESULTS.tsv"
printf 'TR3-P1 candidate PASS: exact zero allocation deltas; retained arena 1024=%s 8192=%s; evidence=%s\n' \
  "${short_bytes}" "${long_bytes}" "${evidence_dir}"
