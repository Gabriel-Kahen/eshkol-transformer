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

evidence="${TR3_PACKAGE_MODE_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-package-mode.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside checkout"

# Documented library-mode object flavor: the Eshkol calling convention is
# retained. A linked --shared-lib instead enables C ABI thunks and currently
# trips the pinned compiler's tail-transfer finalizer on this aggregate.
docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate_root}:/candidate:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/cache
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
/usr/bin/time -v -o /out/package-object.time \
  timeout --foreground --signal=TERM --kill-after=5s 1200s \
  "/candidate/'"${runner_relative}"'" --strict-types --no-stdlib -O 0 \
    -I native -I internal/p1/lib -I internal/c1/lib \
    -I internal/t2/lib -I internal/t1/lib -I internal/d2/lib \
    -I src -I lib --shared-lib -c --emit-depfile /out/private.d \
    native/tr3_c_private_package_root.esk -o /out/private.o \
    > /out/package-object.stdout 2> /out/package-object.stderr
'
[[ -s "${evidence}/private.o" && ! -s "${evidence}/package-object.stderr" ]] || \
  die "library-mode object compilation failed"
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
nm -g --defined-only --format=posix "${evidence}/private.o" | \
  awk '{print $1}' | LC_ALL=C sort -u >"${evidence}/defined-symbols.txt"
nm -u --format=posix "${evidence}/private.o" | \
  awk '{print $1}' | LC_ALL=C sort -u >"${evidence}/undefined-symbols.txt"
cmp native/tr3_c_private_package_mode_defined_symbols.txt "${evidence}/defined-symbols.txt"
cmp native/tr3_c_private_package_mode_undefined_symbols.txt "${evidence}/undefined-symbols.txt"
for symbol in __eshkol_lib_init__ tr3-lease-create-internal \
    tr3-c-trainer-state-internal tr3-c-snapshot-save-internal! \
    tr3-c-checkpoint-restore-internal! tr3-c-trainer-load-state-internal!; do
  grep -Fx -- "${symbol}" "${evidence}/defined-symbols.txt" >/dev/null || \
    die "library-mode object omitted ${symbol}"
done
if grep -E '^(main|et_e1b_public_trainer_.*|.*__eshkol_internal_abi)$' \
    "${evidence}/defined-symbols.txt"; then
  die "private object exposed a program entry, public trainer or C ABI thunk"
fi

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
  -I src -I lib --shared-lib --dump-ir --emit-depfile /out/linked.d \
  native/tr3_c_private_package_root.esk -o /out/linked \
  > /out/linked.stdout 2> /out/linked.stderr'; then
  die "linked package compilation changed; review C ABI boundary"
fi
python3 - "${evidence}/linked.stderr" "${evidence}/linked-tail-failures.txt" <<'PY'
from pathlib import Path
import re
import sys

errors = re.findall(r'Tail transfer: no public entry for ([^\s]+)',
                    Path(sys.argv[1]).read_text())
assert len(errors) == len(set(errors))
Path(sys.argv[2]).write_text(''.join(error + '\n' for error in errors))
PY
cmp native/tr3_c_private_package_mode_linked_tail_failures.txt \
  "${evidence}/linked-tail-failures.txt"
[[ ! -e "${evidence}/linked.ll" && ! -e "${evidence}/liblinked.so" ]] || \
  die "failed linked package compilation published an artifact"

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${candidate_manifest}" runner_sha256)"
  printf 'container_digest\t%s\n' "$(tsv_value "${candidate_manifest}" container_digest)"
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
  printf 'defined_count\t%s\n' "$(wc -l <"${evidence}/defined-symbols.txt")"
  printf 'undefined_count\t%s\n' "$(wc -l <"${evidence}/undefined-symbols.txt")"
  printf 'linked_tail_failures\t%s\n' "$(wc -l <"${evidence}/linked-tail-failures.txt")"
} >"${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C private package-mode object PASS: %s\n' "${evidence}"
printf 'TR3-C private package-mode seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
