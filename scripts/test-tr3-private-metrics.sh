#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in docker git python3 readlink sha256sum; do require_command "${command}"; done
cd "${PROJECT_ROOT}"

pin="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
image="$(tsv_value "${pin}" container_image)"
fixed="${TR3_LINKED_COMPILER_EVIDENCE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/tr3-shared-tail-finalizer-97c40c9d}"
compiler_source="${TR3_LINKED_COMPILER_SOURCE_DIR:-/home/gabe/.codex/worktrees/tr3-shared-tail-finalizer/eshkol}"
runtime="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery}"
case "${TR3_METRICS_RUNTIME_PIN:-81298}" in
  81298)
    runner_name=eshkol-run-release
    runner_sha256=1d4c1a2f6aca335ba873206064e0b3d92d83c457d5dc66f77392e23cc97b47cb
    runtime_build=eshkol-build-canonical
    runtime_sha256="$(tsv_value "${pin}" runtime_archive_sha256)"
    ;;
  fe9)
    runner_name=eshkol-run
    runner_sha256=7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa
    runtime_build=eshkol-build
    runtime_sha256=32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e
    [[ "$(git -C "${compiler_source}" rev-parse HEAD)" == \
       fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9 ]] || die "fe9 compiler source changed"
    [[ "$(git -C "${compiler_source}" rev-parse 'HEAD^{tree}')" == \
       66c21f7ec19b1b4a42199fa30ed8e0e9727021bf ]] || die "fe9 compiler tree changed"
    [[ -z "$(git -C "${compiler_source}" status --porcelain --untracked-files=all)" ]] || \
      die "fe9 compiler source changed"
    [[ "$(tsv_value "${runtime}/${runtime_build}/eshkol-transformer-provenance.tsv" eshkol_commit)" == \
       fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9 ]] || die "fe9 runtime provenance changed"
    ;;
  *) die "unsupported TR3 metrics runtime pin" ;;
esac
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${pin}" container_digest)" ]] || die "pinned image changed"
[[ "$(sha256sum "${fixed}/${runner_name}" | awk '{print $1}')" == \
   "${runner_sha256}" ]] || \
  die "reviewed compiler changed"
[[ "$(sha256sum "${runtime}/${runtime_build}/libeshkol-runtime.a" | awk '{print $1}')" == \
   "${runtime_sha256}" ]] || die "runtime changed"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "metrics gate requires a clean committed checkout"

evidence="${TR3_METRICS_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-private-metrics.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence must be outside checkout"

docker run --rm --network none \
  -e TR3_METRICS_RUNNER="/fixed/${runner_name}" \
  -v "${PROJECT_ROOT}:/workspace:ro" -v "${fixed}:/fixed:ro" \
  -v "${compiler_source}:/fixed-source:ro" \
  -v "${runtime}/${runtime_build}:/runtime/eshkol-build-canonical:ro" \
  -v "${evidence}:/out" \
  -w /workspace "${image}" bash -lc '
set -euo pipefail
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
mkdir -p /out/cache
"${TR3_METRICS_RUNNER}" --strict-types --no-stdlib -O 0 \
  -I native -I lib \
  -L /runtime/eshkol-build-canonical \
  tests/tr3_metrics/private_runtime.esk -o /out/private_runtime \
  > /out/private.compile.stdout 2> /out/private.compile.stderr
"${TR3_METRICS_RUNNER}" --strict-types --no-stdlib -O 0 \
  --compile-only --emit-depfile /out/private.d -I native -I lib \
  tests/tr3_metrics/private_runtime.esk -o /out/private.o \
  > /out/private.deps.stdout 2> /out/private.deps.stderr
ESHKOL_ARENA_POISON=1 /out/private_runtime \
  > /out/private.stdout 2> /out/private.stderr
"${TR3_METRICS_RUNNER}" --strict-types --no-stdlib -O 0 \
  -I native -I lib -L /runtime/eshkol-build-canonical \
  tests/tr3_metrics/failstop_runtime.esk -o /out/failstop_runtime \
  > /out/failstop.compile.stdout 2> /out/failstop.compile.stderr
set +e
ESHKOL_ARENA_POISON=1 /out/failstop_runtime \
  > /out/failstop.stdout 2> /out/failstop.stderr
failstop_status=$?
set -e
test "${failstop_status}" -eq 134
test ! -s /out/failstop.stdout
test ! -s /out/failstop.stderr
printf "failstop_exit\t%s\n" "${failstop_status}" > /out/failstop.tsv
"${TR3_METRICS_RUNNER}" --strict-types --no-stdlib -O 0 --emit-object \
  --emit-depfile /out/allocation.d -I native -I lib \
  tests/tr3_metrics/allocation_probe.esk -o /out/allocation.o \
  > /out/allocation.compile.stdout 2> /out/allocation.compile.stderr
clang++-21 -std=c++17 -O2 -Wall -Wextra -Werror -Wpedantic \
  -I /fixed-source/lib/core -I /fixed-source/inc \
  -c tests/tr3_metrics/allocation_shim.cpp -o /out/allocation_shim.o \
  > /out/shim.compile.stdout 2> /out/shim.compile.stderr
clang++-21 -fPIE -fuse-ld=bfd /out/allocation.o /out/allocation_shim.o \
  /runtime/eshkol-build-canonical/libeshkol-runtime.a \
  -Wl,--wrap=arena_allocate_vector_with_header \
  -Wl,--wrap=_Z28eshkol_region_allocate_quietP5arenamm \
  -Wl,-z,stack-size=536870912 -Wl,--export-dynamic \
  -pthread -ldl -lm -lcrypto -lpng -ljpeg -lwebp -lz -lopenblas \
  -o /out/allocation > /out/link.stdout 2> /out/link.stderr
ESHKOL_ARENA_POISON=1 /out/allocation \
  > /out/allocation.stdout 2> /out/allocation.stderr
'
test ! -s "${evidence}/private.compile.stderr"
test ! -s "${evidence}/private.deps.stderr"
test ! -s "${evidence}/failstop.compile.stderr"
test ! -s "${evidence}/allocation.compile.stderr"
test ! -s "${evidence}/shim.compile.stderr"
test ! -s "${evidence}/link.stderr"
test ! -s "${evidence}/private.stderr"
rg '^TR3-METRICS-PASS checks=42 retained-1024=[0-9]+ retained-8192=[0-9]+ failed-1024=[0-9]+$' \
  "${evidence}/private.stdout" >/dev/null
rg -Fx TR3-METRICS-ALLOCATION-PASS "${evidence}/allocation.stdout" >/dev/null
rg -F 'Failed to allocate vector with header (capacity=10)' \
  "${evidence}/allocation.stderr" >/dev/null
python3 - "${evidence}" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
line = (root / 'private.stdout').read_text().strip()
match = re.fullmatch(
    r'TR3-METRICS-PASS checks=42 retained-1024=(\d+) '
    r'retained-8192=(\d+) failed-1024=(\d+)', line)
assert match is not None
short, long, failed = map(int, match.groups())
assert long - short == 7168 * 336, (short, long)
assert failed > 0
for name in ('private', 'allocation'):
    dep = (root / f'{name}.d').read_text().replace('\\\n', ' ')
    paths = [p.removeprefix('/workspace/') for p in dep.split(':', 1)[1].split()]
    assert 'native/tr3_metrics_extension.esk' in paths, paths
    assert 'lib/transformer/error_internal.esk' in paths, paths
(root / 'retention.tsv').write_text(
    f'published_1024_bytes\t{short}\n'
    f'published_8192_bytes\t{long}\n'
    f'published_incremental_bytes_per_record\t336\n'
    f'failed_1024_arena_bytes\t{failed}\n')
PY
{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'image_digest\t%s\n' "$(tsv_value "${pin}" container_digest)"
  printf 'runtime_pin\t%s\n' "${TR3_METRICS_RUNTIME_PIN:-81298}"
  printf 'runner_sha256\t%s\n' "${runner_sha256}"
  printf 'runtime_archive_sha256\t%s\n' "${runtime_sha256}"
} > "${evidence}/manifest.tsv"
find "${evidence}" -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | (cd "${evidence}" && xargs -d '\n' sha256sum) \
  > "${evidence}/SHA256SUMS"
printf 'TR3 private metrics evidence: %s\n' "${evidence}"
