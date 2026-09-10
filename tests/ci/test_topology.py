"""Exercise the real topology guard against coverage and prerequisite regressions."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TopologyTests(unittest.TestCase):
    def check_topology(self, path=None, before=None, after=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "Makefile",
                "scripts/check-ci-topology.sh",
                "scripts/build.sh",
                ".github/workflows/ci.yml",
                ".github/workflows/acceptance.yml",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            if path:
                target = root / path
                original = target.read_text()
                self.assertIn(before, original)
                target.write_text(original.replace(before, after, 1))
            return subprocess.run(
                ["bash", str(root / "scripts/check-ci-topology.sh")],
                capture_output=True,
                text=True,
            )

    def test_complete_matrix_passes(self):
        result = self.check_topology()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_duplicate_and_full_script_phase_regressions_fail(self):
        command = "\t/usr/bin/bash scripts/test-p1.sh --phase public\n"
        for replacement in (
            "",
            "\t/usr/bin/bash scripts/test-p1.sh --phase state\n",
            "\t/usr/bin/bash scripts/test-p1.sh\n",
        ):
            with self.subTest(replacement=replacement):
                self.assertNotEqual(
                    self.check_topology("Makefile", command, replacement).returncode, 0
                )

    def test_loader_prerequisite_removal_fails(self):
        before = "build-ci-dataset: configure\n\t/usr/bin/bash scripts/build-k1.sh\n"
        self.assertNotEqual(
            self.check_topology("Makefile", before, "build-ci-dataset: configure\n").returncode,
            0,
        )

    def test_serial_acceptance_keeps_full_script(self):
        self.assertNotEqual(
            self.check_topology(
                "Makefile",
                "\t/usr/bin/bash scripts/test-d2.sh\n",
                "\t/usr/bin/bash scripts/test-d2.sh --phase resources\n",
            ).returncode,
            0,
        )

    def test_boundary_phase_cannot_be_omitted_or_replaced_with_full_script(self):
        command = "\t/usr/bin/bash scripts/test-t2-boundary.sh --phase d1-test\n"
        for replacement in ("", "\t/usr/bin/bash scripts/test-t2-boundary.sh\n"):
            with self.subTest(replacement=replacement):
                self.assertNotEqual(
                    self.check_topology("Makefile", command, replacement).returncode, 0
                )

    def test_boundary_caller_requires_canonical_d2(self):
        before = (
            "build-ci-tokenizer-bpe-boundary: configure\n"
            "\t/usr/bin/bash scripts/build-d2.sh\n"
        )
        self.assertNotEqual(
            self.check_topology(
                "Makefile", before, "build-ci-tokenizer-bpe-boundary: configure\n"
            ).returncode,
            0,
        )


if __name__ == "__main__":
    unittest.main()
