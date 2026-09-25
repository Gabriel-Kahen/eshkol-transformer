#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in bash docker git python3 readlink sha256sum mktemp; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "fresh-process gate requires a clean committed checkout"

evidence="${TR3_FRESH_RESUME_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-fresh-resume.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside the source checkout"

python3 - "${PROJECT_ROOT}" "${evidence}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
evidence = Path(sys.argv[2])
base = (root / "tests/tr3_snapshot/runtime_smoke.esk").read_text()
base = base.split('(define tuple (make-tuple 1729))', 1)[0]
base = base.replace(
    '(load "tr3_c_snapshot_root.esk")',
    '(load "tr3_c_private_resume_trajectory_root.esk")',
    1,
)
assert '(load "tr3_c_snapshot_root.esk")' not in base
suffix = (root / "tests/tr3_resume_trajectory/fresh_process_suffix.esk").read_text()
(evidence / "runtime_fresh_resume.esk").write_text(base + "\n" + suffix)
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
visit(root / 'native/tr3_c_private_resume_trajectory_root.esk')
paths = sorted(str(path.relative_to(root)) for path in seen)
for required in ('native/tr3_c_checkpoint_restore_root.esk',
                 'native/tr3_c_snapshot_save_extension.esk',
                 'native/tr3_c_checkpoint_restore_extension.esk',
                 'native/tr3_c_joint_restore_extension.esk',
                 'native/tr3_step_composer_extension.esk',
                 'native/tr3_step_transaction_extension.esk',
                 'native/tr3_step_commit_tail_extension.esk'):
    assert paths.count(required) == 1, required
(evidence / 'source-closure.txt').write_text(''.join(path + '\n' for path in paths))

script = (root / "scripts/test-tr3-c-joint-runtime.sh").read_text()
script = script.replace(
    'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
    f'source "{root}/scripts/common.sh"',
)
script = script.replace("TR3_JOINT_EVIDENCE_DIR", "TR3_FRESH_RESUME_EVIDENCE_DIR")
script = script.replace(
    "tests/tr3_joint_restore/runtime_smoke.esk", "/out/runtime_fresh_resume.esk",
)
script = script.replace("tr3_joint_runtime", "tr3_fresh_resume_runtime")
script = script.replace("joint-runtime", "fresh-resume-runtime")
script = script.replace(
    "compile native/checkpoint_io.c checkpoint_io.o",
    "compile native/checkpoint_io.c checkpoint_io.o -DET_CHECKPOINT_IO_TESTING",
)
script = script.replace(
    "compile native/c2_x1_canonical.c c2_x1_canonical.o",
    "compile native/c2_x1_canonical.c c2_x1_canonical.o\n"
    "compile native/k2_capabilities.c k2_capabilities.o -DET_C2_CARRIER_FACTORIES\n"
    "compile native/c2_checkpoint_codec.c c2_checkpoint_codec.o\n"
    "compile native/c2_checkpoint_format.c c2_checkpoint_format.o\n"
    "compile native/c2_checkpoint_save_bridge.c c2_checkpoint_save_bridge.o\n"
    "compile native/c2_checkpoint_core.c c2_checkpoint_core.o\n"
    "compile native/c2_checkpoint_reader.c c2_checkpoint_reader.o\n"
    "compile native/c2_checkpoint_load_bridge.c c2_checkpoint_load_bridge.o\n"
    "compile native/c2_checkpoint_inspect_bridge.c c2_checkpoint_inspect_bridge.o",
)
runtime = '''ESHKOL_ARENA_POISON=1 /usr/bin/time -v -o /out/runtime.time \\
  timeout --foreground --signal=TERM --kill-after=5s 300s \\
  /out/fresh-resume-runtime /out/corpus \\
  > /out/runtime.stdout 2> /out/runtime.stderr'''
assert runtime in script
script = script.replace(runtime, '''for accumulation in 1 2 3; do
  for mode in uninterrupted producer receiver; do
    ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=5s 300s \\
      /out/fresh-resume-runtime /out/corpus "${mode}" "${accumulation}" \\
      > "/out/${mode}-${accumulation}.stdout" \\
      2> "/out/${mode}-${accumulation}.stderr"
  done
  cmp "/out/corpus/u-${accumulation}-1.c2" "/out/corpus/k-${accumulation}-1.c2"
  cmp "/out/corpus/u-${accumulation}-1.c2" "/out/corpus/r-${accumulation}-0.c2"
  for suffix in 1 2 3; do
    cmp "/out/corpus/u-${accumulation}-$((suffix + 1)).c2" \\
        "/out/corpus/r-${accumulation}-${suffix}.c2"
  done
done''')
runtime_check = '''test ! -s "${evidence}/runtime.stderr"
grep -E '^TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry$' \\
  "${evidence}/runtime.stdout" >/dev/null'''
assert runtime_check in script
script = script.replace(runtime_check, '''for accumulation in 1 2 3; do
  for mode in uninterrupted producer receiver; do
    test ! -s "${evidence}/${mode}-${accumulation}.stderr"
    grep -E "^TR3-C FRESH PROCESS PASS: ${mode} A=${accumulation} checks=[0-9]+$" \\
      "${evidence}/${mode}-${accumulation}.stdout" >/dev/null
  done
done''')
script = script.replace('"${evidence}/runtime.stdout"',
                        '"${evidence}/receiver-3.stdout"')
script = script.replace('"${evidence}/runtime_fresh_resume.esk"',
                        '"${evidence}/runtime_fresh_resume.esk"')
(evidence / "fresh-gate.generated.sh").write_text(script)
PY
chmod +x "${evidence}/fresh-gate.generated.sh"
TR3_FRESH_RESUME_EVIDENCE_DIR="${evidence}" "${evidence}/fresh-gate.generated.sh"

{
  printf 'fresh_gate_sha256\t%s\n' \
    "$(sha256sum scripts/test-tr3-fresh-process-resume.sh | awk '{print $1}')"
  printf 'trajectory_root_sha256\t%s\n' \
    "$(sha256sum native/tr3_c_private_resume_trajectory_root.esk | awk '{print $1}')"
  printf 'fresh_suffix_sha256\t%s\n' \
    "$(sha256sum tests/tr3_resume_trajectory/fresh_process_suffix.esk | awk '{print $1}')"
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
} >>"${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C fresh-process evidence: %s\n' "${evidence}"
printf 'TR3-C fresh-process seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
