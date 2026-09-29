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
old_pin = '''candidate_manifest="${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv"
candidate_root="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-/home/gabe/.codex/evidence/eshkol-transformer/runtime-81298-recovery}"
[[ "${candidate_root}" = /* && -d "${candidate_root}" ]] || \\
  die "frozen TR3 runtime root is unavailable: ${candidate_root}"
candidate_root="$(readlink -f -- "${candidate_root}")"
runner_relative="$(tsv_value "${candidate_manifest}" runner_relative_path)"
runner="${candidate_root}/${runner_relative}"
[[ -x "${runner}" ]] || die "frozen TR3 runner is unavailable"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == \\
   "$(tsv_value "${candidate_manifest}" runner_sha256)" ]] || \\
  die "frozen TR3 runner hash changed"

image="$(tsv_value "${candidate_manifest}" container_image)"'''
new_pin = '''fixed="${TR3_LINKED_COMPILER_EVIDENCE_DIR:-${PROJECT_ROOT}/.deps/eshkol-build}"
compiler_source="${TR3_LINKED_COMPILER_SOURCE_DIR:-${PROJECT_ROOT}/.deps/eshkol-src}"
runtime="${TR3_LEASE_RUNTIME_CANDIDATE_DIR:-${PROJECT_ROOT}/.deps}"
runtime_build_dir="${runtime}/eshkol-build"
compiler_commit=fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9
compiler_tree=66c21f7ec19b1b4a42199fa30ed8e0e9727021bf
runner_sha256=7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa
runtime_sha256=32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e
[[ -x "${fixed}/eshkol-run" && -e "${compiler_source}/.git" &&
   -f "${runtime_build_dir}/libeshkol-runtime.a" ]] ||
  die "fe9 toolchain unavailable"
[[ "$(git -C "${compiler_source}" rev-parse HEAD)" == "${compiler_commit}" &&
   "$(git -C "${compiler_source}" rev-parse 'HEAD^{tree}')" == "${compiler_tree}" &&
   -z "$(git -C "${compiler_source}" status --porcelain --untracked-files=all)" ]] ||
  die "fe9 compiler source changed"
[[ "$(tsv_value "${runtime_build_dir}/eshkol-transformer-provenance.tsv" eshkol_commit)" == \\
   "${compiler_commit}" ]] || die "fe9 runtime provenance changed"
[[ "$(sha256sum "${fixed}/eshkol-run" | awk '{print $1}')" == "${runner_sha256}" &&
   "$(sha256sum "${runtime_build_dir}/libeshkol-runtime.a" | awk '{print $1}')" == \\
   "${runtime_sha256}" ]] || die "fe9 runner or runtime archive changed"

image="$(tsv_value "${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv" container_image)"'''
assert script.count(old_pin) == 1
script = script.replace(old_pin, new_pin)
assert script.count('"${candidate_root}:/candidate:ro"') == 1
script = script.replace('"${candidate_root}:/candidate:ro"',
                        '"${fixed}:/fixed:ro" ' + chr(92) + chr(10) +
                        '  -v "${runtime_build_dir}:/candidate/eshkol-build-canonical:ro"')
assert script.count('/candidate/${runner_relative}') == 1
script = script.replace('/candidate/${runner_relative}', '/fixed/eshkol-run')
assert script.count('-L /out/native --lib tr3_joint_runtime') == 1
script = script.replace('-L /out/native --lib tr3_joint_runtime',
                        '-L /out/native -L /candidate/eshkol-build-canonical --lib tr3_joint_runtime')
script = script.replace('"$(tsv_value "${candidate_manifest}" runner_sha256)"',
                        '"${runner_sha256}"')
script = script.replace('"$(tsv_value "${candidate_manifest}" runner_role)"',
                        '"supported_fe9"')
script = script.replace('"$(tsv_value "${candidate_manifest}" compiler_source_commit)"',
                        '"${compiler_commit}"')
script = script.replace('"$(tsv_value "${candidate_manifest}" container_digest)"',
                        '"$(tsv_value "${PROJECT_ROOT}/tests/tr3_lease/runtime_candidate.tsv" container_digest)"')
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
