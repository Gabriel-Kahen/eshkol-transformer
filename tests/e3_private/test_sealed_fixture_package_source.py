"""Source-only admission controls for the noninstalled sealed-fixture tuple."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build-e1b-consumer.sh"
POLICY = ROOT / "scripts/e3-private-package-policy.sh"
PREFIX = ROOT / "tests/e3_private/sealed_fixture_package"
ROOTS = ["internal/p1/lib", "internal/c1/lib", "internal/t1/lib", "src",
         "internal/d2/lib", "internal/e3/lib"]


def policy(inputs: list[str], roots: list[str] = ROOTS) -> subprocess.CompletedProcess[str]:
    args = [str(ROOT), *inputs, *(str(ROOT / name) for name in roots)]
    script = r'''
set -euo pipefail
PROJECT_ROOT=$1; shift
raw_private_root=$1; raw_package_bridge=$2
raw_package_renames=$3; raw_public_exports=$4; shift 4
raw_include_dirs=("$@")
die() { printf '%s\n' "$*" >&2; exit 1; }
source "${PROJECT_ROOT}/scripts/e3-private-package-policy.sh"
printf '%s\n' "${e3_tuple_kind}"
'''
    return subprocess.run(["/usr/bin/bash", "-c", script, "test", *args],
                          cwd=ROOT, text=True, capture_output=True, check=False)


class PackageSource(unittest.TestCase):
    def test_exact_tuple_and_mixes(self) -> None:
        inputs = [str(PREFIX.parent / "sealed_fixture_driver_root.esk"),
                  str(PREFIX.parent / "sealed_fixture_bridge.c"),
                  str(PREFIX) + "_private_renames.txt",
                  str(PREFIX) + "_public_exports.txt"]
        result = policy(inputs)
        self.assertEqual((result.returncode, result.stdout, result.stderr),
                         (0, "sealed-fixture\n", ""))
        for index, wrong in [(0, ROOT / "tests/e3_private/zero_control_driver_root.esk"),
                             (1, ROOT / "native/e3_private_bridge.c"),
                             (2, ROOT / "tests/e3_private/zero_control_package_private_renames.txt")]:
            bad = inputs.copy()
            bad[index] = str(wrong)
            self.assertNotEqual(policy(bad).returncode, 0)
        self.assertNotEqual(policy(inputs, ROOTS + ["native"]).returncode, 0)

    def test_stage_is_fresh(self) -> None:
        helper = ROOT / "scripts/e3-sealed-fixture-generated-source-stage.sh"
        with tempfile.TemporaryDirectory() as temp:
            gate = Path(temp)
            stage = gate / ".e3-sealed-fixture.ABC123"
            stage.mkdir()
            def attempt(name: str = "e3_sealed_fixture_package.o") -> subprocess.CompletedProcess[str]:
                return subprocess.run(["/usr/bin/bash", "-c",
                                       'source "$1"; e3_sealed_fixture_generated_stage "$2"',
                                       "test", str(helper), str(stage / name)],
                                      text=True, capture_output=True, check=False)
            self.assertEqual(attempt().returncode, 0)
            self.assertNotEqual(attempt("foreign.o").returncode, 0)
            (gate / ".e3-sealed-fixture-generated").mkdir()
            self.assertNotEqual(attempt().returncode, 0)

    def test_three_tu_boundary(self) -> None:
        source = BUILDER.read_text()
        def selected_native(body: str) -> str:
            block = body.split('if [[ "${package_policy}" == e3-private-aggregate &&\n'
                              '          "${e3_tuple_kind}" == sealed-fixture ]]; then\n'
                              '      case "${package_native_source}" in', 1)[1].split('    fi', 1)[0]
            for path in ('m3_call_f32_integration.c', 'e3_frame.c'):
                if block.count(path) != 1:
                    raise ValueError(f"missing or repeated selected TU: {path}")
            if block.count('package_source_flags+=(-DET_G3R_CANDIDATE_RETIRE_PRIVATE') != 1:
                raise ValueError("feature flag was moved or repeated")
            return block
        bridge = source.split('if [[ "${e3_tuple_kind}" == sealed-fixture ]]; then\n'
                              '    package_bridge_flags+=', 1)[1].split('elif [[', 1)[0]
        self.assertIn('(-DET_G3R_CANDIDATE_RETIRE_PRIVATE)', bridge)
        native = selected_native(source)
        self.assertNotIn('ET_G3R_CANDIDATE_RETIRE_PRIVATE', source.replace(bridge, '').replace(native, ''))
        for lost in ('m3_call_f32_integration.c', 'e3_frame.c'):
            with self.assertRaises(ValueError):
                selected_native(source.replace(lost, ''))

    def test_generator_exact_and_mutation(self) -> None:
        tool = ROOT / "scripts/generate-e3-sealed-fixture-source.py"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "overlay"
            argv = ["/usr/bin/python3", "-B", str(tool), "--output-dir", str(target)]
            first = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(subprocess.run(argv + ["--check"], cwd=ROOT,
                                            capture_output=True).returncode, 0)
            for name in ("transformer/module.esk", "i2_wave2_root.esk",
                         "m3t_transport_extension.esk"):
                self.assertTrue((target / name).is_file())
            path = target / "i2_wave2_root.esk"
            path.write_bytes(path.read_bytes() + b"\n")
            self.assertNotEqual(subprocess.run(argv + ["--check"], cwd=ROOT,
                                               capture_output=True).returncode, 0)

    def test_wrapper_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "artifact"
            run = subprocess.run(["/usr/bin/bash", str(ROOT / "scripts/build-e3-sealed-fixture-test.sh"),
                                  str(artifact), "normal"], cwd=ROOT, text=True,
                                 capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("emitted manifest is not admitted", run.stderr)
            self.assertFalse(artifact.exists())


if __name__ == "__main__":
    unittest.main()
