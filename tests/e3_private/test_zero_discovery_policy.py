"""Offline guards for the non-admitting E3 zero inventory discovery route."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build-e1b-consumer.sh"
HARNESS = ROOT / "scripts/discover-e3-zero-inventories.sh"
HELPER = ROOT / "scripts/e3-zero-discovery-evidence.sh"
WORKFLOW = ROOT / ".github/workflows/e3-zero-inventory.yml"


class ZeroDiscoveryPolicyTest(unittest.TestCase):
    def test_exact_deferrals_preserve_authority_and_final_checks(self):
        source = BUILDER.read_text()
        self.assertEqual(source.count('if [[ "${e3_zero_discovery}" != 1 ]]; then'), 3)
        self.assertIn('if [[ "${e3_zero_discovery}" != 1 ]] &&\n'
                      '     ! e1b_compare_exact_undefined_allowlist', source)
        for fragment in (
            'cmp -s "${package_source_closure}" "${e1b_tmp}/source-closure.txt"',
            'cmp -s "${package_native_source_closure}"',
            'cmp -s "${e3_prefix}_sanitized_noninstrumentation_undefined_symbols.txt"',
            'cmp -s "${e3_prefix}_sanitized_instrumentation_undefined_symbols.txt"',
        ):
            self.assertIn(fragment, source)
        terminal = source.index('if [[ "${e3_zero_discovery}" == 1 ]]; then\n'
                                '  e3_zero_discovery_phase=copy-evidence')
        for fragment in ('e3_reject_unresolved_private',
                         'readelf --wide --syms',
                         'E3 required private definition is not local',
                         'zero-p1-postcheck.txt',
                         'object differs from its exact public string manifest'):
            self.assertLess(source.index(fragment), terminal)
        self.assertLess(terminal, source.index('cp "${e1b_tmp}/combined.o" "${temporary_output}"'))
        self.assertIn('cp -a -- "${output_object}.evidence.tmp.$$/." "${dir}/"',
                      HELPER.read_text())
        self.assertIn('e3_zero_discovery_capture failure "${status}"', source)
        policy = (ROOT / "scripts/e3-private-package-policy.sh").read_text()
        self.assertIn('"${E3_ZERO_CONTROL_SANITIZE:-0}" == 1', policy)
        self.assertNotIn('E3_ZERO_CONTROL_DISCOVERY', policy)
        self.assertIn('"${raw_include_dirs[5]}" == "${PROJECT_ROOT}/internal/e3/lib"', source)

    def test_harness_and_workflow_cannot_admit_or_run(self):
        harness = HARNESS.read_text()
        workflow = WORKFLOW.read_text()
        self.assertIn('E3_ZERO_CONTROL_DISCOVERY=1', harness)
        self.assertIn('E3_ZERO_CONTROL_SANITIZE=', harness)
        self.assertIn('[[ ! -e "${object}" && ! -L "${object}"', harness)
        self.assertNotIn('build-e3-zero-control-test.sh', harness)
        self.assertNotIn('test-e3-zero-control.sh', harness)
        for fragment in ('workflow_dispatch:', 'runs-on: ubuntu-22.04',
                         'cancel-in-progress: false', 'make toolchain',
                         'actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02',
                         'if: always()', 'if-no-files-found: error',
                         '6084F3CF814B57C1CF12EFD515CF4D18AF4F7421',
                         "test \"$(llvm-config-21 --version)\" = '21.1.8'"):
            self.assertIn(fragment, workflow)
        self.assertNotIn('pull_request:', workflow)
        self.assertNotIn('test-e3-zero-control.sh', workflow)

    def test_wrong_discovery_marker_rejects_before_toolchain(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner_temp = Path(tmp)
            gate = runner_temp / "e3-n3-zero-discovery/normal"
            stage = gate / ".e3-zero-control.ABC123"
            (gate / "evidence").mkdir(parents=True)
            stage.mkdir()
            prefix = ROOT / "tests/e3_private/zero_control_package"
            args = [ROOT / "tests/e3_private/zero_control_driver_root.esk",
                    ROOT / "tests/e3_private/zero_control_bridge.c",
                    Path(str(prefix) + "_private_renames.txt"),
                    Path(str(prefix) + "_public_exports.txt"),
                    stage / "e3_zero_control_package.o",
                    ROOT / "internal/p1/lib", ROOT / "internal/c1/lib",
                    ROOT / "internal/t1/lib", ROOT / "src",
                    ROOT / "internal/d2/lib", ROOT / "internal/e3/lib"]
            env = dict(os.environ, RUNNER_TEMP=str(runner_temp),
                       E3_ZERO_CONTROL_DISCOVERY="2")
            run = subprocess.run(["bash", str(BUILDER), *map(str, args)],
                                 cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("E3 zero discovery requires its exact tuple", run.stderr)
            self.assertFalse((stage / "e3_zero_control_package.o").exists())

    def test_missing_sanitizer_pins_are_discovery_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner_temp = Path(tmp)
            gate = runner_temp / "e3-n3-zero-discovery/sanitize"
            stage = gate / ".e3-zero-control.ABC123"
            (gate / "evidence").mkdir(parents=True)
            stage.mkdir()
            prefix = ROOT / "tests/e3_private/zero_control_package"
            args = [ROOT / "tests/e3_private/zero_control_driver_root.esk",
                    ROOT / "tests/e3_private/zero_control_bridge.c",
                    Path(str(prefix) + "_private_renames.txt"),
                    Path(str(prefix) + "_public_exports.txt"),
                    stage / "e3_zero_control_package.o",
                    ROOT / "internal/p1/lib", ROOT / "internal/c1/lib",
                    ROOT / "internal/t1/lib", ROOT / "src",
                    ROOT / "internal/d2/lib", ROOT / "internal/e3/lib"]
            env = dict(os.environ, RUNNER_TEMP=str(runner_temp),
                       E3_ZERO_CONTROL_SANITIZE="1",
                       ESHKOL_SOURCE_DIR=str(runner_temp / "absent-source"),
                       ESHKOL_BUILD_DIR=str(runner_temp / "absent-build"))
            self.assertFalse(Path(str(prefix) +
                                  "_sanitized_noninstrumentation_undefined_symbols.txt").exists())
            env["E3_ZERO_CONTROL_DISCOVERY"] = "1"
            discovery = subprocess.run(["bash", str(BUILDER), *map(str, args)],
                                       cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertNotEqual(discovery.returncode, 0)
            self.assertNotIn("sanitized_noninstrumentation_undefined_symbols", discovery.stderr)
            self.assertNotIn("E3 zero discovery arguments differ", discovery.stderr)
            self.assertFalse((stage / "e3_zero_control_package.o").exists())

    def test_capture_copies_partial_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work = base / "work"
            out = base / "evidence"
            generated = base / "generated"
            work.mkdir()
            out.mkdir()
            (generated / "transformer").mkdir(parents=True)
            (work / "private.d").write_text("private: source.esk\n")
            (work / "undefined.txt").write_text("malloc\n")
            (work / "combined.raw.o").write_bytes(b"raw-object")
            (generated / "transformer/module.esk").write_text("(module)\n")
            expected = base / "source-closure.txt"
            expected.write_text("source.esk\n")
            script = f'''set -euo pipefail
source "{ROOT / 'scripts/common.sh'}"
source "{HELPER}"
e1b_tmp="{work}"
e3_zero_discovery_dir="{out}"
e3_generated_source_root="{generated}"
e3_zero_discovery_phase=mock-emission
e3_private_test_sanitize=0
output_object="{base / 'absent.o'}"
mkdir -p "${{output_object}}.evidence.tmp.$$"
printf 'public\n' > "${{output_object}}.evidence.tmp.$$/public-strings.txt"
package_source_closure="{expected}"
package_native_source_closure=/no-native-closure
undefined_symbols=/no-undefined
e3_prefix=/no-prefix
e3_zero_discovery_capture failure 12
'''
            subprocess.run(["bash", "-c", script], check=True, cwd=ROOT)
            self.assertEqual((out / "builder-status.tsv").read_text().splitlines()[0],
                             "result\tfailure")
            self.assertEqual((out / "private.d").read_text(), "private: source.esk\n")
            self.assertEqual((out / "generated/transformer/module.esk").read_text(),
                             "(module)\n")
            self.assertEqual((out / "public-strings.txt").read_text(), "public\n")
            subprocess.run(["sha256sum", "-c", "SHA256SUMS"],
                           check=True, cwd=out, stdout=subprocess.DEVNULL)

    def test_pretoolchain_rejection_still_seals_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner_temp = Path(tmp)
            root = runner_temp / "e3-n3-zero-discovery"
            root.mkdir()
            (root / "status.tsv").write_text("state\treserved\n")
            env = dict(os.environ, RUNNER_TEMP=str(runner_temp),
                       ESHKOL_SOURCE_DIR=str(runner_temp / "absent-source"),
                       ESHKOL_BUILD_DIR=str(runner_temp / "absent-build"))
            run = subprocess.run(["bash", str(HARNESS)], cwd=ROOT, env=env,
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("harness_exit", (root / "status.tsv").read_text())
            subprocess.run(["sha256sum", "-c", "SHA256SUMS"],
                           check=True, cwd=root, stdout=subprocess.DEVNULL)
            self.assertFalse((root / "normal").exists())
