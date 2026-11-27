"""Offline guards for the non-admitting E3 zero inventory discovery route."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from textwrap import dedent
import unittest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build-e1b-consumer.sh"
HARNESS = ROOT / "scripts/discover-e3-zero-inventories.sh"
HELPER = ROOT / "scripts/e3-zero-discovery-evidence.sh"
WORKFLOW = ROOT / ".github/workflows/e3-zero-inventory.yml"


def run_workflow_seal(runner_temp, status):
    workflow = WORKFLOW.read_text()
    seal = workflow.split("      - name: Seal partial or complete discovery evidence\n")[1]
    seal = seal.split("      - name: Upload discovery-only evidence even on failure\n")[0]
    script = dedent(seal.split("        run: |\n")[1])
    env = dict(os.environ, RUNNER_TEMP=str(runner_temp), PREVIOUS_JOB_STATUS=status)
    return subprocess.run(["bash", "-euo", "pipefail", "-c", script], env=env,
                          capture_output=True, text=True)


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
        self.assertIn('cmp "${prefix}_defined_symbols.txt" '
                      '"${gate}/evidence/global-defined.txt"', harness)
        self.assertIn('cmp "${prefix}_native_objects.txt" '
                      '"${gate}/evidence/native-objects.txt"', harness)
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
            checkout = runner_temp / "checkout"
            for directory in ("scripts", "native", "internal", "src", "tests"):
                shutil.copytree(ROOT / directory, checkout / directory)
            # Let the wrapper check its pins, then stop the builder at toolchain
            # verification so this test exercises the discovery route offline.
            with (checkout / "scripts/common.sh").open("a") as common:
                common.write(dedent('''
                    verify_toolchain() {
                      if [[ "${BASH_SOURCE[1]##*/}" == build-e1b-consumer.sh ]]; then
                        die "fixture reached builder toolchain verification"
                      fi
                    }
                '''))
            gate = runner_temp / "e3-n3-zero-discovery/sanitize"
            stage = gate / ".e3-zero-control.ABC123"
            (gate / "evidence").mkdir(parents=True)
            stage.mkdir()
            prefix = checkout / "tests/e3_private/zero_control_package"
            object_file = stage / "e3_zero_control_package.o"
            args = [checkout / "tests/e3_private/zero_control_driver_root.esk",
                    checkout / "tests/e3_private/zero_control_bridge.c",
                    Path(str(prefix) + "_private_renames.txt"),
                    Path(str(prefix) + "_public_exports.txt"),
                    object_file,
                    checkout / "internal/p1/lib", checkout / "internal/c1/lib",
                    checkout / "internal/t1/lib", checkout / "src",
                    checkout / "internal/d2/lib", checkout / "internal/e3/lib"]
            env = dict(os.environ, RUNNER_TEMP=str(runner_temp),
                       E3_ZERO_CONTROL_SANITIZE="1")
            env.pop("E3_ZERO_CONTROL_DISCOVERY", None)
            artifact = runner_temp / "artifact"
            wrapper = checkout / "scripts/build-e3-zero-control-test.sh"
            builder = checkout / BUILDER.relative_to(ROOT)
            for kind in ("noninstrumentation", "instrumentation"):
                with self.subTest(kind=kind):
                    name = f"sanitized_{kind}_undefined_symbols"
                    pin = Path(f"{prefix}_{name}.txt")
                    pin.unlink()
                    normal = subprocess.run(["bash", str(wrapper), str(artifact), "sanitize"],
                                            cwd=checkout, env=env, capture_output=True, text=True)
                    self.assertNotEqual(normal.returncode, 0)
                    self.assertIn(f"sanitizer needs an emitted exact {name} manifest", normal.stderr)
                    self.assertFalse(artifact.exists())
                    discovery = subprocess.run(
                        ["bash", str(builder), *map(str, args)], cwd=checkout,
                        env=dict(env, E3_ZERO_CONTROL_DISCOVERY="1"),
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(discovery.returncode, 0)
                    self.assertIn("fixture reached builder toolchain verification", discovery.stderr)
                    self.assertFalse(object_file.exists())
                    shutil.copyfile(ROOT / pin.relative_to(checkout), pin)

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

    def test_workflow_receipt_survives_artifact_filter_on_success_and_failure(self):
        upload = WORKFLOW.read_text().split(
            "      - name: Upload discovery-only evidence even on failure\n")[1]
        self.assertIn("if: always() && steps.seal-discovery.outcome == 'success'", upload)
        self.assertIn("path: ${{ runner.temp }}/e3-n3-zero-discovery/", upload)
        self.assertIn("include-hidden-files: false", upload)
        sources = ("e3_d2_dataset.esk", "e3_d2_source_provenance.json",
                   "transformer/module.esk")
        for status, modes in (("success", ("normal", "sanitize")),
                              ("failure", ("normal",))):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                runner_temp = Path(tmp)
                root = runner_temp / "e3-n3-zero-discovery"
                root.mkdir()
                (root / "status.tsv").write_text(f"harness_exit\t{int(status == 'failure')}\n")
                (runner_temp / "ambient.txt").write_text("unrelated\n")
                expected = {"status.tsv", "SHA256SUMS"}
                for mode in modes:
                    gate = root / mode
                    stage = gate / ".e3-zero-control.ABC123"
                    stage.mkdir(parents=True)
                    if status == "failure":
                        (stage / "partial.tmp").write_text("builder scratch\n")
                    generated = gate / "evidence/generated"
                    scratch_source = gate / ".e3-zero-control-generated/source"
                    for name in sources:
                        for files in (generated, scratch_source):
                            path = files / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(f"{mode}: {name}\n")
                        expected.add(f"{mode}/evidence/generated/{name}")
                    (gate / "evidence/builder-status.tsv").write_text(f"result\t{status}\n")
                    expected.add(f"{mode}/evidence/builder-status.tsv")

                seal = run_workflow_seal(runner_temp, status)
                self.assertEqual(seal.returncode, 0, seal.stderr)
                scratch, = runner_temp.glob("e3-n3-zero-discovery-scratch.*")
                for mode in modes:
                    self.assertTrue((scratch / mode / ".e3-zero-control.ABC123").is_dir())
                    generated = scratch / mode / ".e3-zero-control-generated/source"
                    for name in sources:
                        self.assertTrue((generated / name).is_file())
                if status == "failure":
                    self.assertEqual(
                        (scratch / "normal/.e3-zero-control.ABC123/partial.tmp").read_text(),
                        "builder scratch\n")

                downloaded = runner_temp / "downloaded"
                shutil.copytree(root, downloaded, ignore=shutil.ignore_patterns(".*"))
                self.assertEqual({str(p.relative_to(downloaded)) for p in downloaded.rglob("*")
                                  if p.is_file()}, expected)
                self.assertIn(f"workflow_status_before_upload\t{status}\n",
                              (downloaded / "status.tsv").read_text())
                manifest = (downloaded / "SHA256SUMS").read_text().splitlines()
                self.assertCountEqual([line.split("  ./", 1)[1] for line in manifest],
                                      expected - {"SHA256SUMS"})
                check = ["sha256sum", "--check", "--strict", "SHA256SUMS"]
                verified = subprocess.run(check, cwd=downloaded, capture_output=True, text=True)
                self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
                omitted = "normal/evidence/generated/transformer/module.esk"
                (downloaded / omitted).unlink()
                broken = subprocess.run(check, cwd=downloaded, capture_output=True, text=True)
                self.assertNotEqual(broken.returncode, 0)
                self.assertIn(omitted, broken.stdout)

    def test_workflow_seal_rejects_hidden_and_nonregular_paths(self):
        for entry in ("normal/evidence/.ambient", "symlink", "pipe"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as tmp:
                runner_temp = Path(tmp)
                root = runner_temp / "e3-n3-zero-discovery"
                root.mkdir()
                (root / "status.tsv").write_text("state\treserved\n")
                path = root / entry
                if entry == "symlink":
                    path.symlink_to(runner_temp / "outside")
                elif entry == "pipe":
                    os.mkfifo(path)
                else:
                    path.parent.mkdir(parents=True)
                    path.write_text("unexpected hidden file\n")
                seal = run_workflow_seal(runner_temp, "failure")
                self.assertNotEqual(seal.returncode, 0)
                self.assertIn(f"Unsupported discovery evidence entry: {path}", seal.stderr)
                self.assertFalse((root / "SHA256SUMS").exists())
