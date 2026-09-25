#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in bash docker python3 readlink sha256sum mktemp; do
  require_command "${command}"
done
cd "${PROJECT_ROOT}"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.tr3_private_load_entry.test_source_contract

evidence="${TR3_LOAD_ENTRY_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-load-entry.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside the source checkout"

# Reuse the accepted joint-restoration native recipe and its original runtime
# witness. The only added source is the private result-cell entry and suffix.
python3 - "${PROJECT_ROOT}" "${evidence}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
evidence = Path(sys.argv[2])
base = (root / "tests/tr3_joint_restore/runtime_smoke.esk").read_text()
original = '(load "tr3_c_joint_restore_extension.esk")'
assert base.count(original) == 1
base = base.replace(
    original,
    original + '\n(load "tr3_c_trainer_load_state_extension.esk")',
)
suffix = (root / "tests/tr3_private_load_entry/runtime_suffix.esk").read_text()
(evidence / "runtime_load_entry.esk").write_text(base + "\n" + suffix)

script = (root / "scripts/test-tr3-c-joint-runtime.sh").read_text()
script = script.replace(
    'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
    f'source "{root}/scripts/common.sh"',
)
script = script.replace("TR3_JOINT_EVIDENCE_DIR", "TR3_LOAD_ENTRY_EVIDENCE_DIR")
script = script.replace(
    "tests/tr3_joint_restore/runtime_smoke.esk", "/out/runtime_load_entry.esk",
)
script = script.replace("tr3_joint_runtime", "tr3_load_entry_runtime")
script = script.replace("joint-runtime", "load-entry-runtime")
script = script.replace(
    "TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry",
    "TR3-C PRIVATE LOAD ENTRY PASS: [0-9]+ checks, result cell, rollback, exact 42-image restore",
)
script = script.replace(
    'sha256sum /out/runtime_load_entry.esk',
    'sha256sum "${evidence}/runtime_load_entry.esk"',
)
(evidence / "load-entry-gate.generated.sh").write_text(script)
PY
chmod +x "${evidence}/load-entry-gate.generated.sh"
TR3_LOAD_ENTRY_EVIDENCE_DIR="${evidence}" \
  "${evidence}/load-entry-gate.generated.sh"

{
  printf 'entry_gate_sha256\t%s\n' \
    "$(sha256sum scripts/test-tr3-c-trainer-load-state.sh | awk '{print $1}')"
  printf 'entry_source_sha256\t%s\n' \
    "$(sha256sum native/tr3_c_trainer_load_state_extension.esk | awk '{print $1}')"
  printf 'entry_suffix_sha256\t%s\n' \
    "$(sha256sum tests/tr3_private_load_entry/runtime_suffix.esk | awk '{print $1}')"
} >>"${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C private load entry evidence: %s\n' "${evidence}"
printf 'TR3-C private load entry seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
