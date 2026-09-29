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
assert base.count("'(none) '(constant)") == 1
base = base.replace("'(none) '(constant)",
                    "'(none) '(linear 2 6 \"3dcccccd\")", 1)
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
script = script.replace('sha256sum /out/runtime_fresh_resume.esk',
                        'sha256sum "${evidence}/runtime_fresh_resume.esk"')
script = script.replace('TR3-C joint runtime evidence:',
                        'TR3-C fresh-process evidence:')
script = script.replace('TR3-C joint runtime seal:',
                        'TR3-C fresh-process seal:')
(evidence / "fresh-gate.generated.sh").write_text(script)
PY
chmod +x "${evidence}/fresh-gate.generated.sh"
TR3_FRESH_RESUME_EVIDENCE_DIR="${evidence}" "${evidence}/fresh-gate.generated.sh"
python3 tests/tr3_resume_trajectory/check_learning_rate.py "${evidence}"

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
