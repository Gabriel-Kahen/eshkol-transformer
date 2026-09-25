#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for command in ar docker git python3 readlink sha256sum; do require_command "${command}"; done
cd "${PROJECT_ROOT}"

candidate="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery}"
manifest="tests/tr3_lease/runtime_candidate.tsv"
image="$(tsv_value "${manifest}" container_image)"
[[ "$(docker image inspect "${image}" --format '{{.Id}}')" == \
   "$(tsv_value "${manifest}" container_digest)" ]] || die "pinned image changed"
[[ "$(sha256sum "${candidate}/eshkol-build-canonical/eshkol-run" | awk '{print $1}')" == \
   "$(tsv_value "${manifest}" runner_sha256)" ]] || die "pinned runner changed"
[[ "$(sha256sum "${candidate}/eshkol-build-canonical/libeshkol-runtime.a" | awk '{print $1}')" == \
   "$(tsv_value "${manifest}" runtime_archive_sha256)" ]] || die "pinned runtime changed"
predecessor="/home/gabe/.codex/evidence/eshkol-transformer/tr3-private-lease-unenroll-linked-de177aa-20260924"
[[ "$(sha256sum "${predecessor}/SHA256SUMS" | awk '{print $1}')" == \
   cbace67a632dd631325ce133beef1047910728b36aad471eac9071d7cf1fb758 ]] || \
   die "accepted native predecessor seal changed"
(cd "${predecessor}" && sha256sum -c SHA256SUMS >/dev/null)
oracle="${TR3_STEP_ORACLE_PYTHON:-${M3_ORACLE_PYTHON:-}}"
[[ "${oracle}" == /* && -x "${oracle}" ]] || die "pinned PyTorch oracle path required"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "step commit-tail gate requires a clean committed checkout"
evidence="${TR3_STEP_TAIL_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-step-tail.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence must be outside the checkout"
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.tr3b.prepare_fixture \
  --output "${evidence}/corpus"

docker run --rm --network none \
  -v "${PROJECT_ROOT}:/workspace:ro" \
  -v "${candidate}:/candidate:ro" \
  -v "${predecessor}:/proof:ro" \
  -v "${evidence}:/out" -w /workspace "${image}" bash -lc '
set -euo pipefail
mkdir -p /out/cache
ar rcsD /out/libtr3_step_native.a \
  /proof/native/data_io.o /proof/native/checkpoint_io.o \
  /proof/native/kernel_abi.o /proof/native/m3_i64_integration.o \
  /proof/native/t1_i64_shell.o /proof/native/m3t_f32_integration.o \
  /proof/native/m3_model.o /proof/native/d2_native.o \
  /proof/native/n2_primitives_provider.o /proof/native/n3k_primitives_provider.o \
  /proof/native/a2_attention_provider.o /proof/native/indexed_cross_entropy.o \
  /proof/native/l3s_masked_objective_provider.o \
  /proof/native/tr3b_objective_bridge.o /proof/native/o2_optimizer.o \
  /proof/native/p1_identity.o /proof/native/i2_bridge.o /proof/native/o2_bridge.o
export ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME=/out/cache
export ESHKOL_LIB_DIR=/workspace/lib ESHKOL_CXX_COMPILER=/usr/bin/clang++-21
/usr/bin/time -v -o /out/compile.time \
  timeout --foreground --signal=TERM --kill-after=5s 900s \
  /candidate/eshkol-build-canonical/eshkol-run --strict-types --no-stdlib -O 0 \
    -I /workspace/native -I /workspace/internal/p1/lib \
    -I /workspace/internal/c1/lib -I /workspace/internal/t2/lib \
    -I /workspace/internal/t1/lib -I /workspace/internal/d2/lib \
    -I /workspace/src -I /workspace/lib -L /out --lib tr3_step_native \
    /workspace/tests/tr3_step_commit_tail/runtime.esk -o /out/runtime \
    > /out/compile.stdout 2> /out/compile.stderr
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 120s \
  /out/runtime /out/corpus > /out/runtime.stdout 2> /out/runtime.stderr
'
test ! -s "${evidence}/compile.stderr"
test ! -s "${evidence}/runtime.stderr"
grep -Fx 'TR3-STEP-TAIL-PASS checks=91' "${evidence}/runtime.stdout" >/dev/null
python3 - "${PROJECT_ROOT}" "${evidence}/source-closure.txt" <<'PY'
from pathlib import Path
import re
import sys
root, output = map(Path, sys.argv[1:])
search = ('native', 'internal/p1/lib', 'internal/c1/lib',
          'internal/t2/lib', 'internal/t1/lib', 'internal/d2/lib',
          'src', 'lib')
seen = set()
def visit(path):
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    for name in re.findall(r'\(load\s+"([^"]+)"\)', path.read_text()):
        found = next((candidate for directory in (path.parent, *(root / item for item in search))
                      if (candidate := directory / name).is_file()), None)
        if found is None:
            raise ValueError(f'missing static load: {name}')
        visit(found)
visit(root / 'tests/tr3_step_commit_tail/runtime.esk')
paths = sorted(str(path.relative_to(root)) for path in seen)
for required in ('native/tr3_step_commit_tail_root.esk',
                 'native/tr3_step_commit_tail_extension.esk',
                 'native/tr3_step_transaction_root.esk',
                 'native/tr3_step_transaction_extension.esk',
                 'native/tr3_step_private_root.esk',
                 'native/tr3_lease_root.esk',
                 'native/tr3b_objective_extension.esk',
                 'native/tr3_o2_step_clear_extension.esk'):
    assert paths.count(required) == 1, required
output.write_text(''.join(path + '\n' for path in paths))
PY
python3 - "${evidence}/runtime.stdout" "${evidence}/numeric-witness.stdout" <<'PYNUM'
from pathlib import Path
import sys
source, target = map(Path, sys.argv[1:])
lines = source.read_text().splitlines()
observations = [line.replace('TR3-STEP-TAIL-OBS', 'TR3-STEP-OBS')
                for line in lines if line.startswith('TR3-STEP-TAIL-OBS ')]
before = next(line for line in lines if line.startswith('TR3-STEP-KEY-BEFORE '))
after = next(line for line in lines if line.startswith('TR3-STEP-KEY-AFTER '))
assert len(observations) == 5 and observations[-2].split()[1] == '0' and observations[-1].split()[1] == '1'
target.write_text('\n'.join((before, *observations[-2:], after,
                             'TR3-STEP-LEAF-PASS checks=41')) + '\n')
PYNUM
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${oracle}" \
  -m tests.tr3_step_leaf.check_numeric "${evidence}/numeric-witness.stdout" \
  > "${evidence}/numeric.stdout" 2> "${evidence}/numeric.stderr"
grep -E '^TR3-STEP-NUMERICAL-PASS before_max_abs=[0-9.e+-]+ after_max_abs=[0-9.e+-]+$' \
  "${evidence}/numeric.stdout" >/dev/null
python3 - "${evidence}/numeric-witness.stdout" "${evidence}/nonfinite.stdout" <<'PYNUM'
from pathlib import Path
import sys
source, target = map(Path, sys.argv[1:])
lines = source.read_text().splitlines()
fields = lines[1].split()
assert fields[:2] == ['TR3-STEP-OBS', '0']
fields[2] = str(0x7fc00000)
lines[1] = ' '.join(fields)
target.write_text('\n'.join(lines) + '\n')
PYNUM
if ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "${oracle}" \
    -m tests.tr3_step_leaf.check_numeric "${evidence}/nonfinite.stdout" \
    > "${evidence}/nonfinite-check.stdout" \
    2> "${evidence}/nonfinite-check.stderr"; then
  die "numeric checker accepted nonfinite objective"
fi
grep -F 'nonfinite objective observation' \
  "${evidence}/nonfinite-check.stderr" >/dev/null
{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'source_tree\t%s\n' "$(git rev-parse 'HEAD^{tree}')"
  printf 'runner_sha256\t%s\n' "$(tsv_value "${manifest}" runner_sha256)"
  printf 'container_digest\t%s\n' "$(tsv_value "${manifest}" container_digest)"
  printf 'predecessor_seal_sha256\t%s\n' cbace67a632dd631325ce133beef1047910728b36aad471eac9071d7cf1fb758
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
  printf 'result\t%s\n' "$(tail -1 "${evidence}/runtime.stdout")"
  printf 'numeric\t%s\n' "$(cat "${evidence}/numeric.stdout")"
  printf 'nonfinite_negative\tPASS\n'
} > "${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum > SHA256SUMS)
printf 'TR3 step commit tail evidence: %s\n' "${evidence}"
printf 'TR3 step commit tail seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
