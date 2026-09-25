#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in cmp docker git nm python3 readlink sha256sum; do require_command "${command}"; done
cd "${PROJECT_ROOT}"

candidate_manifest="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
candidate_root="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery}"
runner_relative="$(tsv_value "${candidate_manifest}" runner_relative_path)"
runner="${candidate_root}/${runner_relative}"
[[ -x "${runner}" && "$(sha256sum "${runner}" | awk '{print $1}')" == \
  "$(tsv_value "${candidate_manifest}" runner_sha256)" ]] || die "pinned runner changed"
image="$(tsv_value "${candidate_manifest}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
  "$(tsv_value "${candidate_manifest}" container_digest)" ]] || die "pinned image changed"

evidence="${TR3_PRIVATE_PACKAGE_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-private-package.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside checkout"

# The compiler's internal-convention object path is an authentic source root
# compilation. It is not a public C-callable package or installed facade.
docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate_root}:/candidate:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/cache
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
/usr/bin/time -v -o /out/object.time \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  "/candidate/'"${runner_relative}"'" --strict-types --no-stdlib -O 0 \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib --emit-object --emit-depfile /out/private.d \
    native/tr3_c_private_package_root.esk -o /out/private.o \
    > /out/object.stdout 2> /out/object.stderr
'
test ! -s "${evidence}/object.stderr"
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
nm -u --format=posix "${evidence}/private.o" | awk '{print $1}' | \
  LC_ALL=C sort -u >"${evidence}/undefined-symbols.txt"
cmp native/tr3_c_private_package_undefined_symbols.txt "${evidence}/undefined-symbols.txt"
nm -g --defined-only --format=posix "${evidence}/private.o" | \
  awk '{print $1}' | LC_ALL=C sort -u >"${evidence}/defined-symbols.txt"
for symbol in tr3-lease-create-internal tr3-c-trainer-state-internal \
    tr3-c-snapshot-save-internal! tr3-c-checkpoint-restore-internal! \
    tr3-c-trainer-load-state-internal!; do
  grep -Fx -- "${symbol}" "${evidence}/defined-symbols.txt" >/dev/null || \
    die "private root omitted ${symbol}"
done
if grep -E '^et_e1b_public_trainer_' "${evidence}/defined-symbols.txt"; then
  die "private object unexpectedly exports a public trainer C symbol"
fi

# Record the exact package-mode compiler blocker on the same source and pin.
if docker run --rm --network none \
    -v "${PROJECT_ROOT}:/workspace:ro" \
    -v "${candidate_root}:/candidate:ro" \
    -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
"/candidate/'"${runner_relative}"'" --strict-types --no-stdlib -O 0 \
  -I native -I internal/p1/lib -I internal/c1/lib \
  -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
  -I src -I lib --shared-lib --dump-ir --emit-depfile /out/shared-lib.d \
  native/tr3_c_private_package_root.esk -o /out/shared-lib \
  > /out/shared-lib.stdout 2> /out/shared-lib.stderr'; then
  die "pinned shared-library compilation unexpectedly succeeded; review package boundary"
fi
grep -F 'Tail transfer: no public entry for tr3-lease-vectors-overlap?__eshkol_tail_body' \
  "${evidence}/shared-lib.stderr" >/dev/null || \
  die "shared-library failure changed; review package boundary"
[[ ! -e "${evidence}/shared-lib.ll" ]] || die "failed package compilation published IR"

# Reuse the accepted checkpoint native recipe and 140-check fixture, changing
# only its root to this aggregate and adding the result-cell witness.
python3 - "${PROJECT_ROOT}" "${evidence}" <<'PY'
from pathlib import Path
import sys

root, evidence = map(Path, sys.argv[1:])
base = (root / 'tests/tr3_snapshot/runtime_smoke.esk').read_text()
assert base.count('(load "tr3_c_snapshot_root.esk")') == 1
base = base.replace('(load "tr3_c_snapshot_root.esk")',
                    '(load "tr3_c_private_package_root.esk")\n'
                    '(define restore-report (k2-public-capability-discover))')
save = (root / 'tests/tr3_snapshot_save/runtime_suffix.esk').read_text()
restore = (root / 'tests/tr3_checkpoint_restore/runtime_suffix.esk').read_text()
suffix = (root / 'tests/tr3_private_package/runtime_suffix.esk').read_text()
(evidence / 'runtime_private_root.esk').write_text(base + '\n' + save + '\n' + restore + '\n' + suffix)
runtime_script = (root / 'scripts/test-tr3-c-joint-runtime.sh').read_text()
runtime_script = runtime_script.replace(
    'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
    f'source "{root}/scripts/common.sh"')
runtime_script = runtime_script.replace('TR3_JOINT_EVIDENCE_DIR',
                                        'TR3_PRIVATE_PACKAGE_EVIDENCE_DIR')
runtime_script = runtime_script.replace('tests/tr3_joint_restore/runtime_smoke.esk',
                                        '/out/runtime_private_root.esk')
runtime_script = runtime_script.replace('tr3_joint_runtime',
                                        'tr3_checkpoint_restore_runtime')
runtime_script = runtime_script.replace('joint-runtime', 'checkpoint-restore-runtime')
runtime_script = runtime_script.replace(
    'compile native/checkpoint_io.c checkpoint_io.o',
    'compile native/checkpoint_io.c checkpoint_io.o -DET_CHECKPOINT_IO_TESTING')
runtime_script = runtime_script.replace(
    'compile native/c2_x1_canonical.c c2_x1_canonical.o',
    'compile native/c2_x1_canonical.c c2_x1_canonical.o\n'
    'compile native/k2_capabilities.c k2_capabilities.o -DET_C2_CARRIER_FACTORIES\n'
    'compile native/c2_checkpoint_codec.c c2_checkpoint_codec.o\n'
    'compile native/c2_checkpoint_format.c c2_checkpoint_format.o\n'
    'compile native/c2_checkpoint_save_bridge.c c2_checkpoint_save_bridge.o\n'
    'compile native/c2_checkpoint_core.c c2_checkpoint_core.o\n'
    'compile native/c2_checkpoint_reader.c c2_checkpoint_reader.o\n'
    'compile native/c2_checkpoint_load_bridge.c c2_checkpoint_load_bridge.o\n'
    'compile native/c2_checkpoint_inspect_bridge.c c2_checkpoint_inspect_bridge.o')
runtime_script = runtime_script.replace(
    'TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry',
    'TR3-C PRIVATE ROOT PASS: [0-9]+ checks, construct/snapshot/SAVE/LOAD/result-cell restore')
runtime_script = runtime_script.replace('sha256sum /out/runtime_private_root.esk',
                                        'sha256sum "${evidence}/runtime_private_root.esk"')
(evidence / 'private-root-runtime.generated.sh').write_text(runtime_script)
PY
chmod +x "${evidence}/private-root-runtime.generated.sh"
TR3_PRIVATE_PACKAGE_EVIDENCE_DIR="${evidence}" \
  "${evidence}/private-root-runtime.generated.sh"

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runner_sha256)"
  printf 'container_digest\t%s\n' "$(tsv_value "${candidate_manifest}" container_digest)"
  printf 'root_sha256\t%s\n' "$(sha256sum native/tr3_c_private_package_root.esk | awk '{print $1}')"
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
  printf 'undefined_count\t%s\n' "$(wc -l <"${evidence}/undefined-symbols.txt")"
  printf 'result\t%s\n' "$(grep '^TR3-C PRIVATE ROOT PASS:' "${evidence}/runtime.stdout")"
} >"${evidence}/private-root-manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C private root evidence: %s\n' "${evidence}"
printf 'TR3-C private root seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
