#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in ar awk cmp docker git grep sha256sum; do require_command "${command}"; done

manifest="${PROJECT_ROOT}/tests/e3_p1/final_runtime_candidate.tsv"
candidate_dir="${E3_P1_RUNTIME_CANDIDATE_DIR:-/tmp/eshkol-rethrow-final-81298b4a-20260923T200234Z}"
runtime_source="${E3_P1_RUNTIME_SOURCE_DIR:-/tmp/eshkol-rethrow-source-81298b4a-20260923T200234Z}"
runner="${candidate_dir}/eshkol-run"
archive="${candidate_dir}/libeshkol-runtime.a"
image="$(tsv_value "${manifest}" container_image)"

[[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=all)" ]] || \
  die "E3-P1 final-runtime evidence requires a clean repository"

[[ -x "${runner}" && -f "${archive}" ]] || \
  die "frozen E3-P1 runtime candidate is unavailable: ${candidate_dir}"
[[ -d "${runtime_source}/.git" ]] || \
  die "frozen E3-P1 runtime source is unavailable: ${runtime_source}"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == \
   "$(tsv_value "${manifest}" runner_sha256)" ]] || \
  die "E3-P1 runtime candidate runner hash changed"
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == \
   "$(tsv_value "${manifest}" runtime_archive_sha256)" ]] || \
  die "E3-P1 runtime candidate archive hash changed"
[[ "$(git -C "${runtime_source}" rev-parse HEAD)" == \
   "$(tsv_value "${manifest}" source_commit)" ]] || \
  die "E3-P1 runtime source commit changed"
[[ "$(git -C "${runtime_source}" rev-parse HEAD^{tree})" == \
   "$(tsv_value "${manifest}" source_tree)" ]] || \
  die "E3-P1 runtime source tree changed"
[[ -z "$(git -C "${runtime_source}" status --porcelain --untracked-files=all)" ]] || \
  die "E3-P1 runtime source is dirty"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${manifest}" container_digest)" ]] || \
  die "E3-P1 supported container identity changed"
grep -Fx "source_commit$(printf '\t')$(tsv_value "${manifest}" source_commit)" \
  "${candidate_dir}/final-provenance.tsv" >/dev/null || \
  die "E3-P1 packaged provenance commit changed"
grep -Fx "source_tree$(printf '\t')$(tsv_value "${manifest}" source_tree)" \
  "${candidate_dir}/final-provenance.tsv" >/dev/null || \
  die "E3-P1 packaged provenance tree changed"

check_source_hash() {
  local key="$1" path="$2"
  [[ "$(sha256sum "${PROJECT_ROOT}/${path}" | awk '{print $1}')" == \
     "$(tsv_value "${manifest}" "${key}")" ]] || \
    die "E3-P1 accepted source hash changed: ${path}"
}
check_source_hash p1_source_sha256 internal/p1/lib/transformer/module.esk
check_source_hash p1_template_sha256 templates/p1/module_roots.esk.tmpl
check_source_hash p1_wrapper_sha256 native/e3_p1_modes_extension.esk
check_source_hash o0_smoke_sha256 tests/e3_p1/modes_smoke.esk
check_source_hash optimized_smoke_sha256 tests/e3_p1/final_runtime.esk
check_source_hash failure_smoke_sha256 tests/e3_p1/final_runtime_failure.esk
check_source_hash test_shim_sha256 tests/e3_p1/final_runtime_shim.cpp
check_source_hash comparator_fixture_sha256 \
  tests/p1/providers/p1_test/tensor_provider.esk

python3 "${PROJECT_ROOT}/scripts/check-e3-p1-contract.py"
"${PROJECT_ROOT}/scripts/generate-p1-roots.sh" --check

evidence_root="${E3_P1_EVIDENCE_ROOT:-$(project_build_dir)}"
mkdir -p "${evidence_root}"
evidence_dir="$(mktemp -d "${evidence_root}/e3-p1-final-runtime.XXXXXX")"
relative_evidence="${evidence_dir#"${PROJECT_ROOT}/"}"
[[ "${relative_evidence}" != "${evidence_dir}" ]] || \
  die "E3-P1 evidence directory must be within the project mount"

git rev-parse HEAD >"${evidence_dir}/repository-commit.txt"
git rev-parse HEAD^{tree} >"${evidence_dir}/repository-tree.txt"
cp "${manifest}" "${evidence_dir}/runtime-candidate.tsv"
cp "${candidate_dir}/final-provenance.tsv" \
  "${evidence_dir}/runtime-final-provenance.tsv"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${candidate_dir}:/candidate:ro" \
  -v "${runtime_source}:/runtime-source:ro" \
  -w /workspace "${image}" bash -lc '
set -euo pipefail
out="$1"
mkdir -p "${out}/native"
exec 3>"${out}/commands.log"
export BASH_XTRACEFD=3
set -x

sha256sum \
  internal/p1/lib/transformer/module.esk \
  templates/p1/module_roots.esk.tmpl \
  native/e3_p1_modes_extension.esk \
  tests/e3_p1/modes_smoke.esk \
  tests/e3_p1/final_runtime.esk \
  tests/e3_p1/final_runtime_failure.esk \
  tests/e3_p1/final_runtime_shim.cpp \
  tests/p1/providers/p1_test/tensor_provider.esk \
  >"${out}/source-hashes.sha256"
sha256sum /candidate/eshkol-run /candidate/libeshkol-runtime.a \
  >"${out}/runtime-hashes.sha256"

cflags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
        -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
        -fexcess-precision=standard -frounding-math
        -I include -I native -I src)
compile_c() {
  local source=$1 object=$2
  shift 2
  clang-21 "${cflags[@]}" "$@" -c "${source}" -o "${out}/native/${object}"
}
compile_c native/a2_attention_provider.c a2_attention_provider.o
compile_c native/checkpoint_io.c checkpoint_io.o
compile_c native/data_io.c data_io.o
compile_c native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD
compile_c native/kernel_abi.c kernel_abi.o
compile_c src/eshkol_transformer/m3_i64_integration.c m3_i64_integration.o
compile_c src/eshkol_transformer/m3_model.c m3_model.o
compile_c src/eshkol_transformer/m3t_f32_integration.c m3t_f32_integration.o \
  -DET_F32_TENSOR_TESTING
compile_c native/n2_primitives_provider.c n2_primitives_provider.o
compile_c native/n3k_primitives_provider.c n3k_primitives_provider.o
compile_c native/p1_identity.c p1_identity.o \
  -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1
compile_c native/t1_i64_shell.c t1_i64_shell.o
clang++-21 -std=c++17 -Wall -Wextra -Werror -fstack-protector-all \
  -fPIC -fvisibility=hidden -fno-common \
  -I /runtime-source/lib/core -I /runtime-source/inc \
  -c tests/e3_p1/final_runtime_shim.cpp \
  -o "${out}/native/final_runtime_shim.o"
ar rcsD "${out}/native/libe3_p1_final_test_runtime.a" "${out}"/native/*.o

includes=(-I native -I internal/p1/lib -I internal/c1/lib -I internal/t1/lib
          -I tests/p1/providers -I src)
link=(-L "${out}/native" --lib e3_p1_final_test_runtime)
export ESHKOL_JIT_CACHE=0 ESHKOL_CXX_COMPILER=/usr/bin/clang++-21

/usr/bin/time -f "compile_peak_rss_kib=%M compile_elapsed_seconds=%e" \
  -o "${out}/o0.compile.time" \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
  /candidate/eshkol-run --strict-types --no-stdlib -O 0 \
    "${includes[@]}" "${link[@]}" tests/e3_p1/modes_smoke.esk \
    -o "${out}/modes-smoke-o0" >"${out}/o0.compile.log" 2>&1
ESHKOL_ARENA_POISON=1 "${out}/modes-smoke-o0" \
  >"${out}/o0.stdout" 2>"${out}/o0.stderr"

/usr/bin/time -f "compile_peak_rss_kib=%M compile_elapsed_seconds=%e" \
  -o "${out}/optimized.compile.time" \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  /candidate/eshkol-run --strict-types --no-stdlib -O 2 --dump-ir \
    "${includes[@]}" "${link[@]}" tests/e3_p1/final_runtime.esk \
    -o "${out}/final-runtime" >"${out}/optimized.compile.log" 2>&1

for horizon in short long; do
  argument=()
  [[ "${horizon}" == short ]] || argument=(long)
  ESHKOL_ARENA_POISON=1 ESHKOL_ARENA_REPORT=1 \
    /usr/bin/time -f "peak_rss_kib=%M elapsed_seconds=%e" \
    "${out}/final-runtime" "${argument[@]}" \
    >"${out}/${horizon}.stdout" 2>"${out}/${horizon}.stderr"
done

/usr/bin/time -f "compile_peak_rss_kib=%M compile_elapsed_seconds=%e" \
  -o "${out}/failure.compile.time" \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  /candidate/eshkol-run --strict-types --no-stdlib -O 2 --dump-ir \
    "${includes[@]}" "${link[@]}" tests/e3_p1/final_runtime_failure.esk \
    -o "${out}/final-runtime-failure" >"${out}/failure.compile.log" 2>&1

for entry in \
  region:0 region:288 region:576 region:864 region:896 region:1056 \
  root:0 root:48 root:208 root:240 root:528 root:816; do
  scope=${entry%%:*}
  budget=${entry##*:}
  E3_P1_FAIL_SCOPE="${scope}" E3_P1_FAIL_BYTES="${budget}" \
    ESHKOL_ARENA_POISON=1 "${out}/final-runtime-failure" \
    >"${out}/failure-${scope}-${budget}.stdout" \
    2>"${out}/failure-${scope}-${budget}.stderr"
done
' bash "${relative_evidence}"

grep -Fx 'E3-P1-MODES-ORDINARY-PASS checks=749' \
  "${evidence_dir}/o0.stdout" >/dev/null
for horizon in 1024 8192; do
  operations=$((horizon * 3))
  file="${evidence_dir}/$([[ "${horizon}" == 1024 ]] && printf short || printf long).stdout"
  grep -Fx 'E3-P1-MODES-ORDINARY-PASS checks=749' "${file}" >/dev/null
  grep -Fx "E3-P1-FINAL-RUNTIME-PASS checks=1075 horizon=${horizon} operations=${operations} arena_used_delta=0 arena_total_delta=0 arena_block_delta=0 native_live_delta=0 native_tombstone_delta=0" \
    "${file}" >/dev/null
done

for budget in 0 288 576 864 896 1056; do
  grep -Fx "E3-P1-BIND-FAILURE-PASS checks=63 budget=${budget} condition=1 used_delta=${budget} total_delta=0 block_delta=0" \
    "${evidence_dir}/failure-region-${budget}.stdout" >/dev/null
done
for budget in 0 48 208 240 528 816; do
  grep -Fx "E3-P1-BIND-FAILURE-PASS checks=63 budget=${budget} condition=2 used_delta=${budget} total_delta=0 block_delta=0" \
    "${evidence_dir}/failure-root-${budget}.stdout" >/dev/null
done

short_total="$(awk -F= '/global_total_allocated_bytes=/{print $2}' \
  "${evidence_dir}/short.stderr" | tail -1)"
long_total="$(awk -F= '/global_total_allocated_bytes=/{print $2}' \
  "${evidence_dir}/long.stderr" | tail -1)"
[[ "${short_total}" =~ ^[0-9]+$ && "${short_total}" == "${long_total}" ]] || \
  die "E3-P1 cumulative root arena differs: 1024=${short_total:-missing} 8192=${long_total:-missing}"
[[ "${short_total}" == "$(tsv_value "${manifest}" expected_root_arena_bytes)" ]] || \
  die "E3-P1 cumulative root arena changed: expected $(tsv_value "${manifest}" expected_root_arena_bytes), got ${short_total}"

short_rss="$(awk -F'[ =]' '/peak_rss_kib=/{print $2}' "${evidence_dir}/short.stderr" | tail -1)"
long_rss="$(awk -F'[ =]' '/peak_rss_kib=/{print $2}' "${evidence_dir}/long.stderr" | tail -1)"
{
  printf 'runtime_commit\t%s\n' "$(tsv_value "${manifest}" source_commit)"
  printf 'runtime_tree\t%s\n' "$(tsv_value "${manifest}" source_tree)"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${manifest}" runner_sha256)"
  printf 'runtime_archive_sha256\t%s\n' "$(tsv_value "${manifest}" runtime_archive_sha256)"
  printf 'short_horizon\t1024\n'
  printf 'long_horizon\t8192\n'
  printf 'short_operations\t3072\n'
  printf 'long_operations\t24576\n'
  printf 'short_root_arena_bytes\t%s\n' "${short_total}"
  printf 'long_root_arena_bytes\t%s\n' "${long_total}"
  printf 'short_peak_rss_kib\t%s\n' "${short_rss}"
  printf 'long_peak_rss_kib\t%s\n' "${long_rss}"
  printf 'constructor_failure_budgets\t0,288,576,864,896,1056\n'
  printf 'promotion_failure_budgets\t0,48,208,240,528,816\n'
} >"${evidence_dir}/summary.tsv"

printf 'E3-P1 final-runtime PASS: O0=749 O2=1075 horizons=1024/8192 root_arena=%s constructor_failpoints=6 promotion_failpoints=6 evidence=%s\n' \
  "${short_total}" "${evidence_dir}"
