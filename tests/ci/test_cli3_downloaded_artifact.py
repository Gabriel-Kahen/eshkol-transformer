"""Benign download fixtures for seal, identity, and safe extraction checks."""

import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "downloaded_artifact", ROOT / "scripts/run-cli3-downloaded-artifact.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class DownloadedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(["git", "-C", str(self.source), "config", "user.email", "fixture@example.test"], check=True)
        subprocess.run(["git", "-C", str(self.source), "config", "user.name", "Fixture"], check=True)
        (self.source / "native").mkdir()
        for name in RUNNER.MANIFESTS:
            (self.source / "native" / f"cli3_generate_{name}.txt").write_text(name + "\n")
        (self.source / "toolchain").mkdir()
        (self.source / "toolchain/eshkol.lock").write_text("fixture-lock\n")
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.source), "commit", "-qm", "fixture"], check=True)
        head = RUNNER.git(self.source, "rev-parse", "HEAD")
        tree = RUNNER.git(self.source, "rev-parse", "HEAD^{tree}")
        self.files = {name: (name + "\n").encode() for name in RUNNER.FILES - {"SHA256SUMS"}}
        self.files["eshkol-transformer"] = b"fixture executable\n"
        for name in RUNNER.MANIFESTS:
            self.files[f"metadata/{name}.txt"] = (name + "\n").encode()
        self.files["metadata/eshkol.lock"] = b"fixture-lock\n"
        self.files["metadata/identity.tsv"] = (
            f"format\teshkol-cli3-binary-v1\nhead\t{head}\ntree\t{tree}\nos\tLinux\narch\tx86_64\n"
        ).encode()
        self.files["metadata/tested-binary.sha256"] = (
            RUNNER.digest(self.files["eshkol-transformer"]) + "  eshkol-transformer\n").encode()
        self.files["metadata/production-generate.pass"] = b"CLI3-PRODUCTION-GENERATE-PASS\n"
        self.files["metadata/cross-command-chain.pass"] = b"CLI3-CROSS-COMMAND-CHAIN-PASS\n"
        self.files["smoke/version.expected"] = b"eshkol-transformer 0.0.1\n"
        self.files["smoke/version.stdout"] = self.files["smoke/version.expected"]
        self.files["smoke/version.stderr"] = b""
        self.files["smoke/tokenizer.stderr"] = b""
        self.download = self.root / "download"
        self.download.mkdir()
        self.archive = self.download / "cli3-binary-artifact.tar"

    def write_archive(self, *, extra=None, reseal_inner=True):
        if reseal_inner:
            self.files["SHA256SUMS"] = b"".join(
                f"{RUNNER.digest(self.files[name])}  ./{name}\n".encode()
                for name in sorted(RUNNER.FILES - {"SHA256SUMS"}))
        with tarfile.open(self.archive, "w", format=tarfile.USTAR_FORMAT) as tar:
            for name in ("cli3-binary-artifact", "cli3-binary-artifact/metadata",
                         "cli3-binary-artifact/smoke"):
                item = tarfile.TarInfo(name)
                item.type = tarfile.DIRTYPE
                tar.addfile(item)
            for name, content in sorted(self.files.items()):
                item = tarfile.TarInfo("cli3-binary-artifact/" + name)
                item.size = len(content)
                item.mode = 0o755 if name == "eshkol-transformer" else 0o644
                tar.addfile(item, io.BytesIO(content))
            if extra:
                tar.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
        (self.download / (self.archive.name + ".sha256")).write_text(
            f"{RUNNER.digest_file(self.archive)}  {self.archive.name}\n")

    def test_accepts_sealed_fixture(self):
        self.write_archive()
        _, _, hashes = RUNNER.verify_archive(self.download, self.source)
        self.assertEqual(hashes["eshkol-transformer"], RUNNER.digest(self.files["eshkol-transformer"]))

    def test_rejects_outer_and_inner_corruption(self):
        self.write_archive()
        with self.archive.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "outer archive SHA-256 mismatch"):
            RUNNER.verify_archive(self.download, self.source)
        self.files["metadata/pretrain.json"] = b"changed\n"
        self.write_archive(reseal_inner=False)
        with self.assertRaisesRegex(ValueError, "inner SHA-256 mismatch"):
            RUNNER.verify_archive(self.download, self.source)

    def test_rejects_tested_binary_gate_and_source_identity(self):
        self.files["metadata/tested-binary.sha256"] = b"0" * 64 + b"  eshkol-transformer\n"
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "tested executable receipt differs"):
            RUNNER.verify_archive(self.download, self.source)
        self.files["metadata/tested-binary.sha256"] = (
            RUNNER.digest(self.files["eshkol-transformer"]) + "  eshkol-transformer\n").encode()
        self.files["metadata/cross-command-chain.pass"] = b"FAIL\n"
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "invalid cross-command-chain gate receipt"):
            RUNNER.verify_archive(self.download, self.source)
        self.files["metadata/cross-command-chain.pass"] = b"CLI3-CROSS-COMMAND-CHAIN-PASS\n"
        self.files["metadata/identity.tsv"] = self.files["metadata/identity.tsv"].replace(b"head\t", b"head\t0")
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "source or platform identity differs"):
            RUNNER.verify_archive(self.download, self.source)

    def test_rejects_traversal_and_symlink_members(self):
        for name, kind in (("cli3-binary-artifact/../escape", tarfile.REGTYPE),
                           ("cli3-binary-artifact/escape", tarfile.SYMTYPE)):
            with self.subTest(name=name, kind=kind):
                extra = tarfile.TarInfo(name)
                extra.type = kind
                extra.size = 1 if kind == tarfile.REGTYPE else 0
                extra.linkname = "../outside" if kind == tarfile.SYMTYPE else ""
                self.write_archive(extra=extra)
                with self.assertRaisesRegex(ValueError, "unsafe or duplicate archive member"):
                    RUNNER.verify_archive(self.download, self.source)
                self.assertFalse((self.root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
