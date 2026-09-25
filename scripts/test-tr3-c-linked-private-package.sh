#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar cmp docker git nm objcopy python3 readelf readlink sha256sum; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"

compiler_evidence="${TR3_LINKED_COMPILER_EVIDENCE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/tr3-shared-tail-finalizer-97c40c9d}"
compiler_source="${TR3_LINKED_COMPILER_SOURCE_DIR:-/home/gabe/.codex/worktrees/tr3-shared-tail-finalizer/eshkol}"
candidate_root="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery}"
candidate_manifest="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
image="$(tsv_value "${candidate_manifest}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
  "$(tsv_value "${candidate_manifest}" container_digest)" ]] || die "pinned image changed"
[[ "$(sha256sum "${compiler_evidence}/eshkol-run-release" | awk '{print $1}')" == \
  1d4c1a2f6aca335ba873206064e0b3d92d83c457d5dc66f77392e23cc97b47cb ]] || \
  die "reviewed fixed compiler changed"
[[ "$(git -C "${compiler_source}" rev-parse HEAD)" == \
  97c40c9de3cf9dfb02a2f5226a14b2a7625b64e0 ]] || \
  die "reviewed fixed compiler headers changed"
git -C "${compiler_source}" diff --quiet HEAD -- \
  inc/eshkol/eshkol.h inc/eshkol/core/runtime.h lib/core/arena_memory.h || \
  die "reviewed fixed compiler headers are modified"
[[ "$(sha256sum "${candidate_root}/eshkol-build-canonical/libeshkol-runtime.a" | awk '{print $1}')" == \
  "$(tsv_value "${candidate_manifest}" runtime_archive_sha256)" ]] || \
  die "pinned runtime archive changed"

evidence="${TR3_LINKED_PACKAGE_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-linked-package.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside checkout"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${compiler_evidence}:/fixed:ro" \
  -v "${compiler_source}:/fixed-source:ro" \
  -v "${candidate_root}:/candidate:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/native /out/cache
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
/usr/bin/time -v -o /out/source-compile.time \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  /fixed/eshkol-run-release --strict-types --no-stdlib -O 0 \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib -L /candidate/eshkol-build-canonical \
    --shared-lib --dump-ir \
    --emit-depfile /out/private.d \
    native/tr3_c_private_package_root.esk -o /out/private \
    > /out/source-compile.stdout 2> /out/source-compile.stderr
test -s /out/private.ll
clang-21 -fPIC -c -x ir /out/private.ll -o /out/private.o

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

clang++-21 -r /out/private.o /out/native/*.o -o /out/combined.raw.o
nm -g --defined-only --format=posix /out/combined.raw.o | \
  awk "{print \$1}" | LC_ALL=C sort -u > /out/raw-defined.txt
awk "\$1 != \"__eshkol_lib_init__\" {print}" /out/raw-defined.txt \
  > /out/localize-symbols.txt
cp /out/combined.raw.o /out/combined.o
objcopy --localize-symbols=/out/localize-symbols.txt /out/combined.o
nm -g --defined-only --format=posix /out/combined.o | \
  awk "{print \$1}" | LC_ALL=C sort -u > /out/global-defined.txt
printf "__eshkol_lib_init__\n" | cmp - /out/global-defined.txt
ar rcsD /out/libtr3_private.a /out/combined.o
cat > /out/tr3.map <<EOF
TR3_PRIVATE_1 { global: __eshkol_lib_init__; local: *; };
EOF
clang++-21 -shared -Wl,--no-undefined -Wl,--version-script=/out/tr3.map \
  -Wl,--whole-archive /out/libtr3_private.a -Wl,--no-whole-archive \
  /candidate/eshkol-build-canonical/libeshkol-runtime.a \
  -lpng -ljpeg -lwebp -lz -lopenblas -lcrypto -pthread -ldl -lm \
  -o /out/libtr3_private.so > /out/link.stdout 2> /out/link.stderr

clang++-21 -std=c++17 -Wall -Wextra -Werror \
  -I /fixed-source/inc -I /fixed-source/lib/core \
  tests/tr3_linked_private_package/init_probe.cpp \
  -Wl,--whole-archive /out/libtr3_private.a -Wl,--no-whole-archive \
  /candidate/eshkol-build-canonical/libeshkol-runtime.a \
  -lpng -ljpeg -lwebp -lz -lopenblas -lcrypto -pthread -ldl -lm \
  -o /out/init-probe
timeout 60s /out/init-probe \
  > /out/init-probe.stdout 2> /out/init-probe.stderr
clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -I /fixed-source/inc -I native \
  native/e1b_error_consumer_bridge.c \
  tests/fixtures/e1b/private_initializer_retry.c \
  -o /out/initializer-retry
timeout 10s /out/initializer-retry \
  > /out/initializer-retry.stdout 2> /out/initializer-retry.stderr

clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic \
  tests/tr3_linked_private_package/dynamic_probe.c -ldl -o /out/dynamic-probe
/out/dynamic-probe /out/libtr3_private.so \
  > /out/dynamic-probe.stdout 2> /out/dynamic-probe.stderr
clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic \
  -c tests/tr3_linked_private_package/link_probe.c -o /out/link-control.o
link_probe() {
  local object=$1 output=$2
  clang++-21 "${object}" -Wl,--whole-archive /out/libtr3_private.a \
    -Wl,--no-whole-archive \
    /candidate/eshkol-build-canonical/libeshkol-runtime.a \
    -lpng -ljpeg -lwebp -lz -lopenblas -lcrypto -pthread -ldl -lm \
    -o "${output}"
}
link_probe /out/link-control.o /out/link-control \
  > /out/link-control.stdout 2> /out/link-control.stderr
/out/link-control
for mode in LEASE NATIVE; do
  clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic \
    "-DTR3_HOSTILE_${mode}" \
    -c tests/tr3_linked_private_package/link_probe.c \
    -o "/out/hostile-${mode}.o"
  if link_probe "/out/hostile-${mode}.o" "/out/hostile-${mode}" \
      > "/out/hostile-${mode}.stdout" \
      2> "/out/hostile-${mode}.stderr"; then
    echo "hostile external link unexpectedly succeeded: ${mode}" >&2
    exit 1
  fi
  test ! -e "/out/hostile-${mode}"
done
grep -F "undefined reference" /out/hostile-LEASE.stderr >/dev/null
grep -F "tr3-lease-create-internal" /out/hostile-LEASE.stderr >/dev/null
grep -F "undefined reference" /out/hostile-NATIVE.stderr >/dev/null
grep -F "et_tr3_c_private_i2_restore_create_v1" \
  /out/hostile-NATIVE.stderr >/dev/null
'

python3 - "${evidence}/private.d" "${evidence}/source-closure.txt" <<'PY'
from pathlib import Path
import sys

dep = Path(sys.argv[1]).read_text().replace('\\\n', ' ')
paths = [path.removeprefix('/workspace/') for path in dep.split(':', 1)[1].split()]
assert paths[0] == 'native/tr3_c_private_package_root.esk'
assert all(path.startswith(('native/', 'internal/', 'src/', 'lib/')) for path in paths)
assert len(paths) == len(set(paths))
Path(sys.argv[2]).write_text(''.join(path + '\n' for path in paths))
PY
cmp native/tr3_c_private_package_source_closure.txt "${evidence}/source-closure.txt"
python3 - "${PROJECT_ROOT}" "${evidence}" <<'PY'
from pathlib import Path
import sys

root, evidence = map(Path, sys.argv[1:])
deps = sorted((evidence / 'native').glob('*.d'))
assert len(deps) == 28
paths = set()
for depfile in deps:
    text = depfile.read_text().replace('\\\n', ' ')
    for item in text.split(':', 1)[1].split():
        path = (root / item).resolve(strict=True)
        assert path.is_relative_to(root), item
        paths.add(path.relative_to(root).as_posix())
(evidence / 'native-source-closure.txt').write_text(
    ''.join(path + '\n' for path in sorted(paths)))
(evidence / 'native-objects.txt').write_text(
    ''.join(path.name[:-2] + '\n' for path in deps))
PY
cmp native/tr3_linked_private_package_native_source_closure.txt \
  "${evidence}/native-source-closure.txt"
cmp native/tr3_linked_private_package_native_objects.txt \
  "${evidence}/native-objects.txt"
nm -u --format=posix "${evidence}/combined.o" | awk '{print $1}' | \
  LC_ALL=C sort -u > "${evidence}/undefined-symbols.txt"
nm -D --defined-only --format=posix "${evidence}/libtr3_private.so" | \
  awk '{print $1}' | LC_ALL=C sort -u > "${evidence}/dynamic-defined.txt"
nm -D --undefined-only --format=posix "${evidence}/libtr3_private.so" | \
  awk '{print $1}' | sed 's/@.*//' | LC_ALL=C sort -u \
  > "${evidence}/dynamic-undefined.txt"
cmp native/tr3_linked_private_package_raw_defined_symbols.txt \
  "${evidence}/raw-defined.txt"
cmp native/tr3_linked_private_package_undefined_symbols.txt \
  "${evidence}/undefined-symbols.txt"
cmp native/tr3_linked_private_package_dynamic_defined_symbols.txt \
  "${evidence}/dynamic-defined.txt"
cmp native/tr3_linked_private_package_dynamic_undefined_symbols.txt \
  "${evidence}/dynamic-undefined.txt"
grep -Fx 'TR3 linked private dynamic boundary PASS' \
  "${evidence}/dynamic-probe.stdout" >/dev/null
grep -E '^TR3 linked private initializer PASS: root_used_before=[0-9]+ root_used_after=[0-9]+$' \
  "${evidence}/init-probe.stdout" >/dev/null
test ! -s "${evidence}/init-probe.stderr"
test ! -s "${evidence}/initializer-retry.stdout"
test ! -s "${evidence}/initializer-retry.stderr"
test ! -s "${evidence}/dynamic-probe.stderr"
test ! -s "${evidence}/link.stderr"
ar t "${evidence}/libtr3_private.a" | \
  cmp native/tr3_linked_private_package_archive_members.txt -
if grep -E '^et_|^tr3-|^c2-' "${evidence}/dynamic-undefined.txt"; then
  die "private package retains unresolved trusted authority"
fi

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'fixed_compiler_commit\t%s\n' 97c40c9de3cf9dfb02a2f5226a14b2a7625b64e0
  printf 'fixed_runner_sha256\t%s\n' 1d4c1a2f6aca335ba873206064e0b3d92d83c457d5dc66f77392e23cc97b47cb
  printf 'container_digest\t%s\n' "$(tsv_value "${candidate_manifest}" container_digest)"
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
  printf 'native_source_count\t%s\n' "$(wc -l <"${evidence}/native-source-closure.txt")"
  printf 'native_object_count\t%s\n' "$(wc -l <"${evidence}/native-objects.txt")"
  printf 'raw_defined_count\t%s\n' "$(wc -l <"${evidence}/raw-defined.txt")"
  printf 'combined_undefined_count\t%s\n' "$(wc -l <"${evidence}/undefined-symbols.txt")"
  printf 'dynamic_export_count\t%s\n' "$(wc -l <"${evidence}/dynamic-defined.txt")"
  printf 'dynamic_boundary\t%s\n' "$(cat "${evidence}/dynamic-probe.stdout")"
  printf 'initializer_boundary\t%s\n' "$(cat "${evidence}/init-probe.stdout")"
} > "${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3 linked private package evidence: %s\n' "${evidence}"
printf 'TR3 linked private package seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
