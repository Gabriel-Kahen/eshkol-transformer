#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in docker git python3 readlink sha256sum; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"

candidate_manifest="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
candidate_root="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery}"
runner="${candidate_root}/eshkol-build-canonical/eshkol-run"
runtime_archive="${candidate_root}/eshkol-build-canonical/libeshkol-runtime.a"
image="$(tsv_value "${candidate_manifest}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
  "$(tsv_value "${candidate_manifest}" container_digest)" ]] || die "pinned image changed"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == \
  "$(tsv_value "${candidate_manifest}" runner_sha256)" ]] || die "pinned runner changed"
[[ "$(sha256sum "${runtime_archive}" | awk '{print $1}')" == \
  "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)" ]] || \
  die "pinned runtime changed"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "unenroll gate requires a clean committed checkout"

evidence="${TR3_UNENROLL_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-unenroll.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside checkout"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate_root}:/candidate:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
export ESHKOL_JIT_CACHE=0 ESHKOL_LIB_DIR=/workspace/lib
base=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
      -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
      -fexcess-precision=standard -frounding-math
      -I include -I native -I src)
for mode in normal sanitize; do
  mkdir -p "/out/${mode}/native" "/out/${mode}/cache"
  extra=()
  if [[ "${mode}" == sanitize ]]; then
    extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    cat > /out/sanitize/cxx <<EOF
#!/usr/bin/env bash
exec /usr/bin/clang++-21 -fsanitize=address,undefined -fno-omit-frame-pointer "\$@"
EOF
    chmod +x /out/sanitize/cxx
    export ESHKOL_CXX_COMPILER=/out/sanitize/cxx
  else
    export ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
  fi
  compile() {
    local source=$1 object=$2
    shift 2
    clang-21 "${base[@]}" "${extra[@]}" "$@" \
      -c "${source}" -o "/out/${mode}/native/${object}"
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
  ar rcsD "/out/${mode}/libtr3_lease_runtime.a" /out/${mode}/native/*.o
  export XDG_CACHE_HOME="/out/${mode}/cache"
  /usr/bin/time -v -o "/out/${mode}/compile.time" \
    timeout --foreground --signal=TERM --kill-after=5s 900s \
    /candidate/eshkol-build-canonical/eshkol-run \
      --strict-types --no-stdlib -O 0 \
      -I tests/tr3_lease -I native -I internal/p1/lib -I internal/c1/lib \
      -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
      -I src -I lib -L "/out/${mode}" --lib tr3_lease_runtime \
      tests/tr3_lease_unenroll/runtime.esk -o "/out/${mode}/runtime" \
      > "/out/${mode}/compile.stdout" 2> "/out/${mode}/compile.stderr"
  horizons=(short long)
  [[ "${mode}" == normal ]] || horizons=(short)
  for horizon in "${horizons[@]}"; do
    args=()
    [[ "${horizon}" == short ]] || args=(long)
    ESHKOL_ARENA_POISON=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
      UBSAN_OPTIONS=halt_on_error=1 \
      timeout --foreground --signal=TERM --kill-after=5s 300s \
      "/out/${mode}/runtime" /out/corpus "${args[@]}" \
      > "/out/${mode}/${horizon}.stdout" \
      2> "/out/${mode}/${horizon}.stderr"
  done
done
'

for mode in normal sanitize; do
  test ! -s "${evidence}/${mode}/compile.stderr"
  for horizon in short long; do
    [[ "${mode}" == normal || "${horizon}" == short ]] || continue
    test ! -s "${evidence}/${mode}/${horizon}.stderr"
    grep -E '^TR3-LEASE-RUNTIME-PASS checks=[0-9]+ horizon=(1024|8192) loop_arena_delta=0 trainers=[0-9]+$' \
      "${evidence}/${mode}/${horizon}.stdout" >/dev/null
    grep -E '^TR3-LEASE-UNENROLL-PASS checks=[0-9]+ trainers=[0-9]+$' \
      "${evidence}/${mode}/${horizon}.stdout" >/dev/null
  done
done
{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runner_sha256)"
  printf 'runtime_archive_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)"
  printf 'container_digest\t%s\n' "$(tsv_value "${candidate_manifest}" container_digest)"
  printf 'normal_short\t%s\n' "$(tail -1 "${evidence}/normal/short.stdout")"
  printf 'normal_long\t%s\n' "$(tail -1 "${evidence}/normal/long.stdout")"
  printf 'sanitize_short\t%s\n' "$(tail -1 "${evidence}/sanitize/short.stdout")"
} > "${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3 private lease unenroll evidence: %s\n' "${evidence}"
printf 'TR3 private lease unenroll seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
