#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in bash docker git python3 readlink sha256sum mktemp; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "retention-cycle gate requires a clean committed checkout"

evidence="${TR3_RETENTION_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-retention.XXXXXX")}"
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
    '(load "tr3_c_retention_cycle_root.esk")',
    1,
)
assert '(load "tr3_c_snapshot_root.esk")' not in base
definitions = (root / "tests/tr3_resume_trajectory/fresh_process_suffix.esk").read_text()
definitions = definitions.split('(check "supported accumulation profile"', 1)[0]
suffix = (root / "tests/tr3_resume_trajectory/retention_cycle_suffix.esk").read_text()
(evidence / "runtime_retention.esk").write_text(base + "\n" + definitions + "\n" + suffix)
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
visit(root / 'native/tr3_c_retention_cycle_root.esk')
paths = sorted(str(path.relative_to(root)) for path in seen)
for required in ('native/tr3_c_snapshot_save_extension.esk',
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
script = script.replace("TR3_JOINT_EVIDENCE_DIR", "TR3_RETENTION_EVIDENCE_DIR")
script = script.replace(
    "tests/tr3_joint_restore/runtime_smoke.esk", "/out/runtime_retention.esk",
)
script = script.replace("tr3_joint_runtime", "tr3_retention_runtime")
script = script.replace("joint-runtime", "retention-runtime")
script = script.replace(
    '  -w /workspace "${image}" bash -lc',
    '  -e "TR3_CYCLE_HORIZONS=${TR3_CYCLE_HORIZONS:-32 128}" '
    '-w /workspace "${image}" bash -lc',
)
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
script = script.replace(
    'compile native/p1_identity.c p1_identity.o -DET_P1_TRUSTED_BUILD=1',
    'compile native/p1_identity.c p1_identity.o -DET_P1_TRUSTED_BUILD=1 -DET_P1_TEST_HOOKS=1')
script = script.replace(
    'compile native/c2_checkpoint_core.c c2_checkpoint_core.o',
    'compile native/c2_checkpoint_core.c c2_checkpoint_core.o -DET_C2_CHECKPOINT_CORE_TESTING')
script = script.replace(
    'compile native/tr3_c_restore_bindings.c tr3_c_restore_bindings.o',
    'compile tests/tr3_resume_trajectory/retention_counts.c cycle_counts.o '
    '-DET_F32_TENSOR_TESTING -DET_O2_TESTING\n'
    'compile native/tr3_c_restore_bindings.c tr3_c_restore_bindings.o')
script = script.replace(
    '       -I include -I native -I src)',
    '       -I include -I native -I src\n'
    '       -fsanitize=address,undefined -fno-omit-frame-pointer)')
script = script.replace(
    'export ESHKOL_CXX_COMPILER=/usr/bin/clang++-21',
    'printf "#!/bin/sh\nexec /usr/bin/clang++-21 -fsanitize=address,undefined '
    '-fno-omit-frame-pointer \\\"\\$@\\\"\n" > /out/clang++-san\n'
    'chmod +x /out/clang++-san\n'
    'export ESHKOL_CXX_COMPILER=/out/clang++-san')
runtime = '''ESHKOL_ARENA_POISON=1 /usr/bin/time -v -o /out/runtime.time \\
  timeout --foreground --signal=TERM --kill-after=5s 300s \\
  /out/retention-runtime /out/corpus \\
  > /out/runtime.stdout 2> /out/runtime.stderr'''
assert runtime in script
script = script.replace(runtime, '''for horizon in ${TR3_CYCLE_HORIZONS:-32 128}; do
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \\
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \\
  ESHKOL_ARENA_POISON=1 ESHKOL_ARENA_REPORT=1 \\
    /usr/bin/time -v -o "/out/${horizon}.time" \\
    timeout --foreground --signal=TERM --kill-after=5s 1200s \\
      /out/retention-runtime /out/corpus cycle 2 "${horizon}" \\
      > "/out/${horizon}.stdout" 2> "/out/${horizon}.stderr"
done''')
check = '''test ! -s "${evidence}/runtime.stderr"
grep -E '^TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry$' \\
  "${evidence}/runtime.stdout" >/dev/null'''
assert check in script
script = script.replace(check, '''for horizon in ${TR3_CYCLE_HORIZONS:-32 128}; do
  grep -E "^TR3-C CYCLE PASS horizon=${horizon} checks=[0-9]+ arena_delta=[0-9]+$" \\
    "${evidence}/${horizon}.stdout" >/dev/null
  if grep -E 'ERROR: AddressSanitizer|runtime error:|LeakSanitizer' \\
      "${evidence}/${horizon}.stderr"; then exit 1; fi
done''')
script = script.replace('"$(cat "${evidence}/runtime.stdout")"',
                        '"cycle horizons ${TR3_CYCLE_HORIZONS:-32 128} passed"')
script = script.replace('sha256sum /out/runtime_retention.esk',
                        'sha256sum "${evidence}/runtime_retention.esk"')
script = script.replace('TR3-C joint runtime evidence:', 'TR3-C retention evidence:')
script = script.replace('TR3-C joint runtime seal:', 'TR3-C retention seal:')
(evidence / 'retention-gate.generated.sh').write_text(script)
PY
chmod +x "${evidence}/retention-gate.generated.sh"
TR3_RETENTION_EVIDENCE_DIR="${evidence}" "${evidence}/retention-gate.generated.sh"

{
  printf 'retention_gate_sha256\t%s\n' \
    "$(sha256sum scripts/test-tr3-retention-cycle.sh | awk '{print $1}')"
  printf 'retention_root_sha256\t%s\n' \
    "$(sha256sum native/tr3_c_retention_cycle_root.esk | awk '{print $1}')"
  printf 'retention_suffix_sha256\t%s\n' \
    "$(sha256sum tests/tr3_resume_trajectory/retention_cycle_suffix.esk | awk '{print $1}')"
  printf 'source_count\t%s\n' "$(wc -l <"${evidence}/source-closure.txt")"
} >>"${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C retention evidence: %s\n' "${evidence}"
printf 'TR3-C retention seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
