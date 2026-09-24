#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for command in bash python3 readlink sha256sum mktemp; do
  require_command "${command}"
done

cd "${PROJECT_ROOT}"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests.tr3_snapshot.test_source_contract \
  tests.tr3_lease.test_snapshot_authority_contract

evidence="${TR3_SNAPSHOT_COMPOSER_EVIDENCE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tr3-snapshot-composer.XXXXXX")}"
mkdir -p -- "${evidence}"
evidence="$(readlink -f -- "${evidence}")"
[[ "${evidence}" != "${PROJECT_ROOT}" && "${evidence}" != "${PROJECT_ROOT}/"* ]] || \
  die "evidence directory must be outside the source checkout"

# The accepted joint gate owns the supported f31/LLVM21 native build recipe.
# Generate a bounded sibling that changes only the fixture, archive/binary
# names, evidence variable, and expected result. This prevents a second native
# build policy from drifting away from the already reviewed aggregate recipe.
generated="${evidence}/snapshot-composer-gate.generated.sh"
python3 - "${PROJECT_ROOT}" "${generated}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
source = (root / "scripts/test-tr3-c-joint-runtime.sh").read_text()
source = source.replace(
    'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
    f'source "{root}/scripts/common.sh"',
)
source = source.replace("TR3_JOINT_EVIDENCE_DIR", "TR3_SNAPSHOT_COMPOSER_EVIDENCE_DIR")
source = source.replace(
    "tests/tr3_joint_restore/runtime_smoke.esk",
    "tests/tr3_snapshot/runtime_smoke.esk",
)
source = source.replace("tr3_joint_runtime", "tr3_snapshot_runtime")
source = source.replace("joint-runtime", "snapshot-runtime")
source = source.replace(
    "TR3-C JOINT RUNTIME PASS: [0-9]+ checks, 42 tensors, abort/retry",
    "TR3-C SNAPSHOT RUNTIME PASS: [0-9]+ checks, 42 tensors, transfer/abort/retry",
)
source = source.replace("TR3-C joint runtime", "TR3-C snapshot composer runtime")
output.write_text(source)
PY
chmod +x "${generated}"
TR3_SNAPSHOT_COMPOSER_EVIDENCE_DIR="${evidence}" "${generated}"

{
  printf 'composer_gate_sha256\t%s\n' \
    "$(sha256sum scripts/test-tr3-c-snapshot-composer.sh | awk '{print $1}')"
  printf 'inherited_native_gate_sha256\t%s\n' \
    "$(sha256sum scripts/test-tr3-c-joint-runtime.sh | awk '{print $1}')"
  printf 'snapshot_source_sha256\t%s\n' \
    "$(sha256sum native/tr3_c_snapshot_extension.esk | awk '{print $1}')"
} >>"${evidence}/manifest.tsv"
(cd "${evidence}" && find . -type f ! -name SHA256SUMS -printf '%P\n' | \
  LC_ALL=C sort | xargs sha256sum >SHA256SUMS)
printf 'TR3-C snapshot composer final evidence: %s\n' "${evidence}"
printf 'TR3-C snapshot composer final seal: %s\n' \
  "$(sha256sum "${evidence}/SHA256SUMS" | awk '{print $1}')"
