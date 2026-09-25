#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in bash docker python3 readlink sha256sum mktemp; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.tr3_checkpoint_restore.test_source_contract

evidence="${TR3_CHECKPOINT_RESTORE_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-checkpoint-restore.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside the source checkout"

python3 - "${PROJECT_ROOT}" "${evidence}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
evidence = Path(sys.argv[2])
base = (root / "tests/tr3_snapshot/runtime_smoke.esk").read_text()
base = base.replace(
    '(load "tr3_c_snapshot_root.esk")',
    '(load "tr3_c_checkpoint_restore_root.esk")\n'
    '(define restore-report (k2-public-capability-discover))',
    1,
)
assert '(load "tr3_c_snapshot_root.esk")' not in base
save_suffix = (root / "tests/tr3_snapshot_save/runtime_suffix.esk").read_text()
restore_suffix = (root / "tests/tr3_checkpoint_restore/runtime_suffix.esk").read_text()
(evidence / "runtime_restore.esk").write_text(base + "\n" + save_suffix + "\n" + restore_suffix)

script = (root / "scripts/test-tr3-c-joint-runtime.sh").read_text()
script = script.replace(
    'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
    f'source "{root}/scripts/common.sh"',
)
script = script.replace("TR3_JOINT_EVIDENCE_DIR", "TR3_CHECKPOINT_RESTORE_EVIDENCE_DIR")
script = script.replace(
    "tests/tr3_joint_restore/runtime_smoke.esk", "/out/runtime_restore.esk",
)
script = script.replace("tr3_joint_runtime", "tr3_checkpoint_restore_runtime")
script = script.replace("joint-runtime", "checkpoint-restore-runtime")
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
    "TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry",
    "TR3-C CHECKPOINT RESTORE PASS: [0-9]+ checks, C2 LOAD and exact live restore",
)
script = script.replace(
    'sha256sum /out/runtime_restore.esk',
    'sha256sum "${evidence}/runtime_restore.esk"',
)
(evidence / "restore-gate.generated.sh").write_text(script)
PY
chmod +x "${evidence}/restore-gate.generated.sh"
TR3_CHECKPOINT_RESTORE_EVIDENCE_DIR="${evidence}" "${evidence}/restore-gate.generated.sh"

{
  printf 'restore_gate_sha256\t%s\n' \
    "$(sha256sum scripts/test-tr3-c-checkpoint-restore.sh | awk '{print $1}')"
  printf 'checkpoint_restore_source_sha256\t%s\n' \
    "$(sha256sum native/tr3_c_checkpoint_restore_extension.esk | awk '{print $1}')"
  printf 'restore_suffix_sha256\t%s\n' \
    "$(sha256sum tests/tr3_checkpoint_restore/runtime_suffix.esk | awk '{print $1}')"
} >>"${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C checkpoint restore evidence: %s\n' "${evidence}"
printf 'TR3-C checkpoint restore seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
