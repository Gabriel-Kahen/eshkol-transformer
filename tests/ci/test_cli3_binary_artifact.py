"""Reject artifact custody errors before any native build or host verification."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "scripts/package-cli3-binary.sh"


class CLI3BinaryArtifactTests(unittest.TestCase):
    def run_rejected(self, *args: Path) -> None:
        result = subprocess.run(
            ["/usr/bin/bash", str(PACKAGE), *(str(arg) for arg in args)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse("CLI3-BINARY-ARTIFACT-PASS" in result.stdout)

    def test_rejects_missing_or_relative_paths(self) -> None:
        self.run_rejected()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_rejected(root, Path("relative-artifact"))
            self.assertFalse((root / "relative-artifact").exists())

    def test_rejects_preexisting_or_symlinked_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
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
            evidence = root / "evidence"
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

    def test_rejects_evidence_inside_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.run_rejected(ROOT / "tests/ci", Path(temporary) / "artifact")


if __name__ == "__main__":
    unittest.main()
