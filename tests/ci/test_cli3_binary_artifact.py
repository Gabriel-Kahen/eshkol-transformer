"""Reject artifact custody errors before any native build or host verification."""

from pathlib import Path
import hashlib
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "scripts/package-cli3-binary.sh"
CROSS_COMMAND = ROOT / "scripts/test-cli3-cross-command-chain.sh"


class CLI3BinaryArtifactTests(unittest.TestCase):
    def run_rejected(self, *args: Path) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["/usr/bin/bash", str(PACKAGE), *(str(arg) for arg in args)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse("CLI3-BINARY-ARTIFACT-PASS" in result.stdout)
        return result

    def test_rejects_missing_or_relative_paths(self) -> None:
        self.run_rejected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_rejected(root, Path("relative-artifact"))
            self.assertFalse((root / "relative-artifact").exists())

    def test_rejects_preexisting_or_symlinked_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "production-generate"
            evidence.mkdir()
            artifact = root / "artifact"
            artifact.mkdir()
            self.run_rejected(evidence, artifact)
            artifact.rmdir()
            artifact.symlink_to(evidence, target_is_directory=True)
            self.run_rejected(evidence, artifact)
            evidence_alias = root / "evidence-alias"
            evidence_alias.symlink_to(evidence, target_is_directory=True)
            self.run_rejected(evidence_alias, root / "new-artifact")
            self.assertFalse((root / "new-artifact").exists())

            archive = root / "archive.tar"
            archive.write_bytes(b"must not replace")
            self.run_rejected(evidence, root / "archive")
            self.assertEqual(archive.read_bytes(), b"must not replace")

    def test_rejects_escaped_or_symlinked_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "production-generate"
            evidence.mkdir()
            artifact = root / "artifact"
            outside = root / "outside"
            outside.mkdir()
            (evidence / "scratch-path.txt").write_text(str(outside) + "\n")
            self.run_rejected(evidence, artifact)
            (evidence / "scratch-path.txt").write_text(str(evidence / ".build.fake") + "\n")
            (evidence / ".build.fake").symlink_to(outside, target_is_directory=True)
            self.run_rejected(evidence, artifact)
            self.assertFalse(artifact.exists())

    def test_rejects_artifact_inside_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "production-generate"
            evidence.mkdir()
            self.run_rejected(evidence, ROOT / "tests/ci/new-artifact")

    def test_rejects_nonproduction_evidence_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            result = self.run_rejected(evidence, Path(temporary) / "artifact")
            self.assertIn("production-generate evidence child", result.stderr)

    def test_rejects_tampered_or_redirected_pretest_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "production-generate"
            scratch = evidence / ".build.fake"
            scratch.mkdir(parents=True)
            binary = scratch / "eshkol-transformer"
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            (scratch / "config.json").write_text("{}\n")
            (scratch / "byte.t1").write_bytes(b"fixture")
            (evidence / "tokenizer.json").write_text("{}\n")
            (evidence / "scratch-path.txt").write_text(str(scratch) + "\n")
            receipt = evidence / "binary-before.sha256"
            receipt.write_text("0" * 64 + "  " + str(binary) + "\n")
            result = self.run_rejected(evidence, root / "artifact")
            self.assertIn("differs from its pre-test receipt", result.stderr)
            other = root / "other"
            other.write_bytes(binary.read_bytes())
            receipt.write_text(hashlib.sha256(other.read_bytes()).hexdigest()
                               + "  " + str(other) + "\n")
            result = self.run_rejected(evidence, root / "artifact")
            self.assertIn("differs from its pre-test receipt", result.stderr)
            self.assertFalse((root / "artifact").exists())

    def test_cross_command_rejects_incomplete_or_changed_production_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            production = root / "production-generate"
            build = production / ".build.fixture"
            build.mkdir(parents=True)
            cross = root / "cross-command-chain"
            binary = build / "eshkol-transformer"
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            (production / "scratch-path.txt").write_text(str(build) + "\n")
            (production / "binary-before.sha256").write_text(
                "0" * 64 + "  " + str(binary) + "\n")
            command = ["/usr/bin/bash", str(CROSS_COMMAND), str(cross), str(production)]

            for expected in ("requires a passed production gate", "differs from its pre-test receipt"):
                with self.subTest(expected=expected):
                    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
                    self.assertFalse((cross / "gate.pass").exists())
                (production / "gate.pass").write_text("CLI3-PRODUCTION-GENERATE-PASS\n")


if __name__ == "__main__":
    unittest.main()
