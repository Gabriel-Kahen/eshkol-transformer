#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar docker git python3 sha256sum; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"

base_commit=c342891277a935a1105f8e3e641311e35d3abf45
base="${TR3_STOP_POLICY_BASE_EVIDENCE:-/home/gabe/.codex/evidence/eshkol-transformer/tr3-stop-policy-public-20260928}"
fixed="${TR3_LINKED_COMPILER_EVIDENCE_DIR:-/home/gabe/.codex/worktrees/wave3-runtime-repin/eshkol-transformer/.deps/eshkol-build}"
compiler_source="${TR3_LINKED_COMPILER_SOURCE_DIR:-/home/gabe/.codex/worktrees/wave3-runtime-repin/eshkol-transformer/.deps/eshkol-src}"
runtime="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/worktrees/wave3-runtime-repin/eshkol-transformer/.deps}"
evidence="${TR3_STOP_POLICY_RETENTION_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-stop-policy-retention.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] ||
  die "evidence must be outside checkout"
[[ "$(awk -F '\t' '$1 == "source_commit" {print $2}' "${base}/manifest.tsv")" == "${base_commit}" ]] ||
  die "base archive has wrong source commit"
[[ "$(sha256sum "${base}/SHA256SUMS" | awk '{print $1}')" == \
   79bf4a3f1e27bd731559cac7c079fac687376cfd507479d09885d8a7f9733c1c ]] ||
  die "base evidence seal changed"
(cd "${base}" && sha256sum -c SHA256SUMS >/dev/null)
git diff --quiet "${base_commit}" -- lib/transformer/trainer.esk \
  native/tr3_stop_policy_extension.esk native/tr3_public_installed_root.esk \
  native/tr3_public_installed_bridge.c native/tr3_public_installed_exports.txt \
  native/tr3_public_installed_private_renames.txt ||
  die "production stop-policy source differs from sealed base"
[[ "$(git -C "${compiler_source}" rev-parse HEAD)" == \
   fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9 ]] ||
  die "pinned compiler source differs"
[[ "$(sha256sum "${fixed}/eshkol-run" | awk '{print $1}')" == \
   7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa ]] ||
  die "pinned runner differs"
[[ "$(sha256sum "${runtime}/eshkol-build/libeshkol-runtime.a" | awk '{print $1}')" == \
   32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e ]] ||
  die "pinned runtime differs"
image=sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6
docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" -v "${base}:/prior:ro" \
  -v "${fixed}:/fixed:ro" -v "${compiler_source}:/fixed-source:ro" \
  -v "${runtime}/eshkol-build:/candidate/eshkol-build-canonical:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/cache
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache ESHKOL_LIB_DIR=/prior/facades
clang-21 -std=c11 -Wall -Wextra -Werror -Wpedantic -fPIC \
  -I /fixed-source/inc -I native -c tests/tr3_metrics/linked_producer.c \
  -o /out/linked_producer.o
ar rcsD /out/libeshkol_transformer_tr3_metrics_test.a \
  /prior/combined.raw.o /out/linked_producer.o
timeout --foreground --signal=TERM --kill-after=5s 120s \
  /fixed/eshkol-run --strict-types --no-stdlib -O 0 \
    -I /prior/facades -L /out -L /candidate/eshkol-build-canonical \
    --lib eshkol_transformer_tr3_metrics_test \
    tests/tr3_public_installed/stop_policy_retention.esk -o /out/retention \
    > /out/compile.stdout 2> /out/compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/retention > /out/run.stdout 2> /out/run.stderr
'
test ! -s "${evidence}/compile.stderr"
test ! -s "${evidence}/run.stderr"
python3 - "${evidence}" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
line = (root / 'run.stdout').read_text().strip()
match = re.fullmatch(r'TR3-STOP-POLICY-RETENTION-PASS '
                     r'retained-1023=(\d+) retained-7168=(\d+)', line)
assert match is not None, line
short, long = map(int, match.groups())
assert short == 1023 * 224, (short, long)
assert long == 7168 * 224, (short, long)
(root / 'retention.tsv').write_text(
    f'published_1023_bytes\t{short}\n'
    f'published_next_7168_bytes\t{long}\n'
    'incremental_bytes_per_policy\t224\n'
    'oldest_policy_after_8192\tvalidated\n'
    'authority_record_shape\t6 slots: tag,self,opaque-token,three scalars\n')
PY
{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'base_commit\t%s\n' "${base_commit}"
  printf 'base_archive_sha256\t%s\n' "$(sha256sum "${base}/combined.raw.o" | awk '{print $1}')"
  printf 'fixed_runner_sha256\t%s\n' "$(sha256sum "${fixed}/eshkol-run" | awk '{print $1}')"
  printf 'container_digest\t%s\n' "${image}"
} > "${evidence}/manifest.tsv"
find "${evidence}" -type f ! -name SHA256SUMS -printf '%P\n' |
  LC_ALL=C sort | (cd "${evidence}" && xargs -d '\n' sha256sum) > "${evidence}/SHA256SUMS"
printf 'TR3 stop-policy retention PASS: %s\n' "${evidence}"
