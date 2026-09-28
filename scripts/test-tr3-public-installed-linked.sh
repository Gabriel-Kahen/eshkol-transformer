#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar cmp docker git nm objcopy python3 readlink rg sha256sum; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"

fixed="${TR3_LINKED_COMPILER_EVIDENCE_DIR:-${PROJECT_ROOT}/.deps/eshkol-build}"
compiler_source="${TR3_LINKED_COMPILER_SOURCE_DIR:-${PROJECT_ROOT}/.deps/eshkol-src}"
runtime="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-${PROJECT_ROOT}/.deps}"
pin="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
[[ -f "${fixed}/eshkol-run" && -e "${compiler_source}/.git" &&
   -f "${runtime}/eshkol-build/libeshkol-runtime.a" ]] ||
  die "fe9 toolchain unavailable; set TR3_LINKED_COMPILER_EVIDENCE_DIR, TR3_LINKED_COMPILER_SOURCE_DIR, and TR3_LEASE_RUNTIME_CANDIDATE_DIR"
case "${TR3_PUBLIC_INSTALLED_RUNTIME_PIN:-fe9}" in
  fe9)
    compiler_commit=fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9
    runner_name=eshkol-run
    compiler_tree=66c21f7ec19b1b4a42199fa30ed8e0e9727021bf
    runner_sha256=7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa
    runtime_build_dir="${runtime}/eshkol-build"
    runtime_sha256=32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e
    source_manifest=native/tr3_public_installed_fe9_source_closure.txt
    undefined_manifest=native/tr3_public_installed_fe9_undefined_symbols.txt
    [[ "$(git -C "${compiler_source}" rev-parse 'HEAD^{tree}')" == "${compiler_tree}" ]] || \
      die "fe9 compiler tree changed"
    [[ -z "$(git -C "${compiler_source}" status --porcelain --untracked-files=all)" ]] || \
      die "fe9 compiler source changed"
    [[ "$(tsv_value "${runtime_build_dir}/eshkol-transformer-provenance.tsv" eshkol_commit)" == \
       "${compiler_commit}" ]] || die "fe9 runtime provenance changed"
    ;;
  81298) die "public metrics-ref requires the fe9 true-f32 runtime" ;;
  *) die "unsupported TR3 public installed runtime pin" ;;
esac
image="$(tsv_value "${pin}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${pin}" container_digest)" ]] || die "pinned image changed"
[[ "$(sha256sum "${fixed}/${runner_name}" | awk '{print $1}')" == \
   "${runner_sha256}" ]] || \
  die "reviewed compiler changed"
[[ "$(git -C "${compiler_source}" rev-parse HEAD)" == \
   "${compiler_commit}" ]] || \
  die "reviewed compiler source changed"
git -C "${compiler_source}" diff --quiet HEAD -- \
  inc/eshkol/eshkol.h inc/eshkol/core/runtime.h lib/core/arena_memory.h || \
  die "reviewed compiler headers changed"
[[ "$(sha256sum "${runtime_build_dir}/libeshkol-runtime.a" | awk '{print $1}')" == \
   "${runtime_sha256}" ]] || \
  die "pinned runtime archive changed"

evidence="${TR3_PUBLIC_INSTALLED_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-public-installed.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence must be outside checkout"

docker run --rm --network none \
  -e TR3_COMPILER_RUNNER="/fixed/${runner_name}" \
  -v "${PROJECT_ROOT}:/workspace:ro" -v "${fixed}:/fixed:ro" \
  -v "${compiler_source}:/fixed-source:ro" \
  -v "${runtime_build_dir}:/candidate/eshkol-build-canonical:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/native /out/cache
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
/usr/bin/time -v -o /out/source-compile.time \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib -L /candidate/eshkol-build-canonical \
    --shared-lib --dump-ir --emit-depfile /out/private.d \
    native/tr3_public_installed_root.esk -o /out/private \
    > /out/source-compile.stdout 2> /out/source-compile.stderr
test -s /out/private.ll
clang-21 -fPIC -c -x ir /out/private.ll -o /out/private.o
objcopy --redefine-syms=native/tr3_public_installed_private_renames.txt \
  /out/private.o

flags=(-std=c11 -Wall -Wextra -Werror -Wpedantic -fstack-protector-all
       -fPIC -fvisibility=hidden -fno-common -ffp-contract=off
       -fexcess-precision=standard -frounding-math
       -I include -I native -I src)
compile() {
  local source=$1 object=$2
  shift 2
  clang-21 "${flags[@]}" "$@" -MMD -MF "/out/native/${object}.d" \
    -c "${source}" -o "/out/native/${object}"
}
compile native/data_io.c data_io.o
compile native/checkpoint_io.c checkpoint_io.o
compile native/kernel_abi.c kernel_abi.o
compile src/eshkol_transformer/m3_i64_integration.c m3_i64_integration.o
compile native/t1_i64_shell.c t1_i64_shell.o
compile src/eshkol_transformer/m3t_f32_integration.c m3t_f32_integration.o \
  -DET_I2_PRIVATE_OWNED_CLONE_MATCH
compile src/eshkol_transformer/m3_model.c m3_model.o
compile native/d2_native.c d2_native.o -DET_TR3_C_D2_RESTORE
compile native/n2_primitives_provider.c n2_primitives_provider.o
compile native/n3k_primitives_provider.c n3k_primitives_provider.o
compile native/a2_attention_provider.c a2_attention_provider.o
compile native/indexed_cross_entropy.c indexed_cross_entropy.o
compile native/l3s_masked_objective_provider.c l3s_masked_objective_provider.o
compile native/tr3b_objective_bridge.c tr3b_objective_bridge.o
compile native/o2_optimizer.c o2_optimizer.o \
  -DET_TR3_O2_STEP_CLEAR_NATIVE -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_O2_RESTORE_NATIVE
compile native/p1_identity.c p1_identity.o -DET_P1_TRUSTED_BUILD=1
compile native/i2_wave2_package_bridge.c i2_bridge.o \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_M3T_PACKAGE_BUILD \
  -DET_C2_I2_MODEL_COPY -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE
compile native/o2_wave2_package_bridge.c o2_bridge.o \
  -DET_O2_NATIVE_HELPERS_ONLY -DET_C2_O2_RECONSTRUCT_BRIDGE \
  -DET_TR3_O2_STEP_CLEAR_BRIDGE
compile native/c2_x1_canonical.c c2_x1_canonical.o
compile native/k2_capabilities.c k2_capabilities.o -DET_C2_CARRIER_FACTORIES
compile native/c2_checkpoint_codec.c c2_checkpoint_codec.o
compile native/c2_checkpoint_format.c c2_checkpoint_format.o
compile native/c2_checkpoint_save_bridge.c c2_checkpoint_save_bridge.o
compile native/c2_checkpoint_core.c c2_checkpoint_core.o
compile native/c2_checkpoint_reader.c c2_checkpoint_reader.o
compile native/c2_checkpoint_load_bridge.c c2_checkpoint_load_bridge.o
compile native/c2_checkpoint_inspect_bridge.c c2_checkpoint_inspect_bridge.o
compile native/tr3_c_restore_bindings.c tr3_c_restore_bindings.o \
  -DET_TR3_C_RESTORE_BINDINGS -DET_I2_PRIVATE_OWNED_CLONE_MATCH \
  -DET_TR3_C_I2_RESTORE_PRIVATE -DET_TR3_C_O2_RESTORE_NATIVE
compile native/tr3_public_step_metrics.c tr3_public_step_metrics.o
clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -I /fixed-source/inc -I native -MMD -MF /out/native/e1b_bridge.o.d \
  -c native/e1b_error_consumer_bridge.c -o /out/native/e1b_bridge.o
clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -I /fixed-source/inc -I native -MMD -MF /out/native/tr3_public_installed_bridge.o.d \
  -c native/tr3_public_installed_bridge.c \
  -o /out/native/tr3_public_installed_bridge.o

clang++-21 -r /out/private.o /out/native/*.o -o /out/combined.raw.o
nm -g --defined-only --format=posix /out/combined.raw.o | \
  awk "{print \$1}" | LC_ALL=C sort -u > /out/raw-defined.txt
awk "NR == FNR {allowed[\$1] = 1; next} !(\$1 in allowed) {print \$1}" \
  native/tr3_public_installed_exports.txt /out/raw-defined.txt \
  > /out/localize-symbols.txt
cp /out/combined.raw.o /out/combined.o
objcopy --localize-symbols=/out/localize-symbols.txt /out/combined.o
nm -g --defined-only --format=posix /out/combined.o | \
  awk "{print \$1}" | LC_ALL=C sort -u > /out/global-defined.txt
cmp native/tr3_public_installed_exports.txt /out/global-defined.txt
nm -u --format=posix /out/combined.o | awk "{print \$1}" | \
  LC_ALL=C sort -u > /out/undefined-symbols.txt
ar rcsD /out/libeshkol_transformer_tr3_public_installed.a /out/combined.o
test "$(ar t /out/libeshkol_transformer_tr3_public_installed.a)" = combined.o

mkdir -p /out/corpus/installed-d1
mkdir -p /out/facades/transformer
while IFS= read -r facade; do
  test -f "lib/${facade}" && test ! -L "lib/${facade}"
  cp "lib/${facade}" "/out/facades/${facade}"
done < native/tr3_public_installed_facades.txt
export ESHKOL_LIB_DIR=/out/facades
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades --compile-only --emit-depfile /out/caller.d \
    tests/tr3_public_installed/runtime.esk -o /out/caller.o \
    > /out/caller-check.stdout 2> /out/caller-check.stderr
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_public_installed \
    tests/tr3_public_installed/runtime.esk \
    -o /out/caller > /out/caller-compile.stdout 2> /out/caller-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/caller /out/corpus/installed-d1 \
  > /out/caller.stdout 2> /out/caller.stderr
grep -Fx TR3-PUBLIC-OPERANDS-PASS /out/caller.stdout >/dev/null
test ! -s /out/caller.stderr

# The installed package intentionally has no public step producer yet. Link a
# test-only producer to the unlocalized copy of this exact aggregate so its
# private authority can publish one authentic record for the public reader.
clang-21 "${flags[@]}" -I /fixed-source/inc \
  -c tests/tr3_metrics/linked_producer.c -o /out/linked_producer.o
ar rcsD /out/libeshkol_transformer_tr3_metrics_test.a \
  /out/combined.raw.o /out/linked_producer.o
timeout --foreground --signal=TERM --kill-after=5s 120s \
  "${TR3_COMPILER_RUNNER}" --strict-types --no-stdlib -O 0 \
    -I /out/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_metrics_test \
    tests/tr3_metrics/linked_runtime.esk -o /out/linked_metrics_caller \
    > /out/linked-metrics-compile.stdout 2> /out/linked-metrics-compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/linked_metrics_caller \
  > /out/linked-metrics.stdout 2> /out/linked-metrics.stderr
grep -Fx TR3-PUBLIC-METRICS-LINKED-PASS /out/linked-metrics.stdout >/dev/null
test ! -s /out/linked-metrics.stderr

# Focused bit and counter boundary proof for the actual installed C bridge.
clang-21 "${flags[@]}" -ffunction-sections -fdata-sections \
  -I /fixed-source/inc -c native/tr3_public_installed_bridge.c \
  -o /out/metrics_bridge_unit.o
clang-21 "${flags[@]}" -I /fixed-source/inc \
  -c tests/tr3_metrics/public_bridge.c -o /out/metrics_bridge_unit_test.o
clang++-21 -Wl,--gc-sections /out/metrics_bridge_unit.o \
  /out/metrics_bridge_unit_test.o \
  /candidate/eshkol-build-canonical/libeshkol-runtime.a \
  -pthread -ldl -lm -o /out/metrics_bridge_unit_test
/out/metrics_bridge_unit_test > /out/metrics-bridge.stdout \
  2> /out/metrics-bridge.stderr
grep -Fx TR3-PUBLIC-METRICS-BRIDGE-PASS /out/metrics-bridge.stdout >/dev/null
test ! -s /out/metrics-bridge.stderr
'

python3 - "${evidence}/private.d" "${evidence}/source-closure.txt" <<'PY'
from pathlib import Path
import sys
dep = Path(sys.argv[1]).read_text().replace('\\\n', ' ')
paths = [path.removeprefix('/workspace/') for path in dep.split(':', 1)[1].split()]
assert len(paths) == len(set(paths))
Path(sys.argv[2]).write_text(''.join(path + '\n' for path in paths))
PY
cmp "${source_manifest}" "${evidence}/source-closure.txt"
python3 - "${PROJECT_ROOT}" "${evidence}" <<'PY'
from pathlib import Path
import sys
root, evidence = map(Path, sys.argv[1:])
deps = sorted((evidence / 'native').glob('*.d'))
assert len(deps) == 31
paths = set()
for dep in deps:
    text = dep.read_text().replace('\\\n', ' ')
    for item in text.split(':', 1)[1].split():
        if item.startswith('/workspace/'):
            item = item.removeprefix('/workspace/')
        elif item.startswith('/'):
            continue
        path = (root / item).resolve(strict=True)
        assert path.is_relative_to(root), item
        paths.add(path.relative_to(root).as_posix())
(evidence / 'native-source-closure.txt').write_text(
    ''.join(path + '\n' for path in sorted(paths)))
(evidence / 'native-objects.txt').write_text(
    ''.join(dep.name.removesuffix('.d') + '\n' for dep in deps))
PY
cmp native/tr3_public_installed_native_source_closure.txt \
  "${evidence}/native-source-closure.txt"
cmp native/tr3_public_installed_native_objects.txt \
  "${evidence}/native-objects.txt"
while IFS= read -r facade; do
  cmp "lib/${facade}" "${evidence}/facades/${facade}"
done < native/tr3_public_installed_facades.txt
cmp "${undefined_manifest}" \
  "${evidence}/undefined-symbols.txt"
python3 - "${evidence}/caller.d" <<'PY'
from pathlib import Path
import sys
dep = Path(sys.argv[1]).read_text().replace('\\\n', ' ')
paths = [path.removeprefix('/workspace/') for path in dep.split(':', 1)[1].split()]
assert paths == ['tests/tr3_public_installed/runtime.esk',
                 '/out/facades/transformer/config.esk',
                 '/out/facades/transformer/error_consumer.esk',
                 '/out/facades/transformer/data.esk',
                 '/out/facades/transformer/diagnostic_transport.esk',
                 '/out/facades/transformer/module.esk',
                 '/out/facades/transformer/optim.esk',
                 '/out/facades/transformer/tokenizer.esk',
                 '/out/facades/transformer/trainer.esk'], paths
PY
if rg '^et_e1b_private_|^tr3-lease-create-internal$|^tr3-lease-unenroll-internal!$|^trainer-create$|^trainer-release!$|^trainer-step!$|^c2-public-trainer-state-release!$' \
    "${evidence}/global-defined.txt"; then
  die "candidate package leaked private trainer authority"
fi
{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'fixed_compiler_commit\t%s\n' "${compiler_commit}"
  printf 'fixed_runner_sha256\t%s\n' "${runner_sha256}"
  printf 'container_digest\t%s\n' "$(tsv_value "${pin}" container_digest)"
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
  printf 'global_defined_count\t%s\n' "$(wc -l <"${evidence}/global-defined.txt")"
  printf 'undefined_count\t%s\n' "$(wc -l <"${evidence}/undefined-symbols.txt")"
} >"${evidence}/manifest.tsv"
find "${evidence}" -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | (cd "${evidence}" && xargs -d '\n' sha256sum) \
  >"${evidence}/SHA256SUMS"
printf 'TR3 public installed linked PASS: %s\n' "${evidence}"
