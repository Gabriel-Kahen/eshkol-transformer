"""Lightweight CLI3 gate evidence checks; these never invoke a compiler."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-cli3-corpus-validate.sh"


class EvidenceDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cli3-evidence-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_gate(self, directory: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), str(directory)],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            env=env,
        )

    def test_occupied_path_is_rejected_without_mutation(self) -> None:
        occupied = self.root / "occupied"
        occupied.mkdir()
        sentinel = occupied / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        result = self.run_gate(occupied)
        self.assertEqual(result.returncode, 64, result.stdout + result.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertEqual([path.name for path in occupied.iterdir()], ["sentinel"])
        self.assertIn("evidence directory already exists", result.stderr)

    def test_prebuild_failure_retains_logs_and_exit(self) -> None:
        evidence = self.root / "failed-run"
        env = os.environ.copy()
        env["ESHKOL_SOURCE_DIR"] = str(self.root / "missing-source")
        env["ESHKOL_BUILD_DIR"] = str(self.root / "missing-build")
        result = self.run_gate(evidence, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(evidence.is_dir())
        steps = evidence / "steps"
        self.assertEqual(
            int((steps / "verify-toolchain.status").read_text(encoding="utf-8")),
            result.returncode,
        )
        self.assertEqual(
            int((evidence / "overall.status").read_text(encoding="utf-8")),
            result.returncode,
        )
        self.assertTrue((steps / "verify-toolchain.stdout").is_file())
        self.assertTrue((steps / "verify-toolchain.stderr").is_file())
        self.assertGreater((steps / "verify-toolchain.stderr").stat().st_size, 0)
        self.assertFalse((steps / "build-d1.status").exists())
        self.assertFalse((evidence / "artifacts" / "eshkol-transformer").exists())


if __name__ == "__main__":
    unittest.main()
