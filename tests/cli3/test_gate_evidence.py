"""Lightweight CLI3 gate evidence checks; these never invoke a compiler."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-cli3-corpus-validate.sh"
PROJECT = SCRIPT.parents[1]


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

    def test_checkout_paths_are_rejected_before_creation(self) -> None:
        alias = self.root / "checkout-alias"
        alias.symlink_to(PROJECT, target_is_directory=True)
        for candidate in (
            PROJECT / f".cli3-evidence-test-{os.getpid()}",
            alias / f".cli3-evidence-test-{os.getpid()}",
        ):
            with self.subTest(candidate=candidate):
                self.assertFalse(candidate.exists())
                result = self.run_gate(candidate)
                self.assertEqual(result.returncode, 64, result.stdout + result.stderr)
                self.assertIn("outside the project checkout", result.stderr)
                self.assertFalse(candidate.exists())
        self.assertEqual(
            subprocess.check_output(
                ["git", "-C", str(PROJECT), "status", "--porcelain"], text=True
            ),
            "",
        )

    def test_prebuild_failure_retains_logs_and_exit(self) -> None:
        evidence = self.root / "failed-run"
        env = os.environ.copy()
        env["ESHKOL_SOURCE_DIR"] = str(self.root / "missing-source")
        env["ESHKOL_BUILD_DIR"] = str(self.root / "missing-build")
        result = self.run_gate(evidence, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(evidence.is_dir())
        steps = evidence / "steps"
        metadata = dict(
            line.split("\t", 1)
            for line in (evidence / "metadata.tsv").read_text(encoding="utf-8").splitlines()
        )
        head = subprocess.check_output(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        self.assertEqual(metadata["project_head"], head)
        self.assertEqual(metadata["project_tree"], tree)
        self.assertEqual(metadata["project_clean"], "true")
        self.assertEqual(
            metadata["lock_sha256"],
            hashlib.sha256((PROJECT / "toolchain" / "eshkol.lock").read_bytes()).hexdigest(),
        )
        self.assertEqual(metadata["compiler_sha256"], "missing")
        self.assertEqual(metadata["compiler_provenance_commit"], "missing")
        self.assertEqual((evidence / "initial-git-status.txt").read_text(), "")
        self.assertEqual((evidence / "final-project-head.txt").read_text().strip(), head)
        self.assertEqual((evidence / "final-project-tree.txt").read_text().strip(), tree)
        self.assertEqual((evidence / "final-git-status.txt").read_text(), "")
        self.assertEqual((steps / "capture-metadata.status").read_text().strip(), "0")
        self.assertEqual((steps / "check-project-pin.status").read_text().strip(), "0")
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

    def test_dirty_index_is_rejected_before_toolchain_without_repo_mutation(self) -> None:
        evidence = self.root / "dirty-index"
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(self.root / "absent-alternate-index")
        result = self.run_gate(evidence, env)
        self.assertNotEqual(result.returncode, 0)
        metadata = dict(
            line.split("\t", 1)
            for line in (evidence / "metadata.tsv").read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(metadata["project_clean"], "false")
        self.assertGreater((evidence / "initial-git-status.txt").stat().st_size, 0)
        self.assertNotEqual(
            (evidence / "steps" / "check-project-pin.status").read_text().strip(), "0"
        )
        self.assertFalse((evidence / "steps" / "verify-toolchain.status").exists())
        self.assertFalse((self.root / "absent-alternate-index").exists())
        self.assertEqual(
            subprocess.check_output(
                ["git", "-C", str(PROJECT), "status", "--porcelain"], text=True
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
