"""Benign download fixtures for seal, identity, and safe extraction checks."""

import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock


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
            if extra is not None:
                tar.addfile(extra, io.BytesIO(b"x" * extra.size) if extra.size else None)
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
        (self.download / (self.archive.name + ".sha256")).write_text(
            f"{RUNNER.digest_file(self.archive)}  {self.archive.name}\n")

    def test_accepts_sealed_fixture(self):
        self.write_archive()
        _, _, hashes = RUNNER.verify_archive(self.download, self.source)
        self.assertEqual(hashes["eshkol-transformer"], RUNNER.digest(self.files["eshkol-transformer"]))

    def set_build_commit(self, build_commit):
        identity = self.files["metadata/identity.tsv"].splitlines(keepends=True)
        identity[1] = f"head\t{build_commit}\n".encode()
        self.files["metadata/identity.tsv"] = b"".join(identity)

    def test_accepts_distinct_head_with_the_exact_build_tree(self):
        self.write_archive()
        build_commit = RUNNER.git(self.source, "rev-parse", "HEAD")
        RUNNER.git(self.source, "commit", "--allow-empty", "-qm", "same tree")
        self.assertNotEqual(RUNNER.git(self.source, "rev-parse", "HEAD"), build_commit)
        RUNNER.verify_archive(self.download, self.source)

    def test_accepts_unavailable_synthetic_commit_with_the_exact_tree(self):
        self.set_build_commit("f" * 40)
        self.write_archive()
        RUNNER.verify_archive(self.download, self.source)

    def test_rejects_a_checkout_with_a_different_tree(self):
        self.write_archive()
        (self.source / "extra").write_text("another source file\n")
        RUNNER.git(self.source, "add", "extra")
        RUNNER.git(self.source, "commit", "-qm", "different tree")
        with self.assertRaisesRegex(ValueError, "source or platform identity differs"):
            RUNNER.verify_archive(self.download, self.source)

    def test_rejects_a_locally_known_inconsistent_build_commit(self):
        empty_tree = RUNNER.git(self.source, "mktree", input_text="")
        wrong_commit = RUNNER.git(self.source, "commit-tree", empty_tree, "-m", "different tree")
        for object_id in (wrong_commit, empty_tree):
            with self.subTest(object_id=object_id):
                self.set_build_commit(object_id)
                self.write_archive()
                with self.assertRaisesRegex(ValueError, "build commit does not identify the recorded tree"):
                    RUNNER.verify_archive(self.download, self.source)

    def test_uses_the_same_archive_for_outer_hash_and_member_verification(self):
        self.write_archive()
        replacement = self.root / "replacement.tar"
        replacement.write_bytes(b"different archive")
        real_digest_stream = RUNNER.digest_stream
        replaced = False

        def swap_after_first_digest(stream):
            nonlocal replaced
            digest = real_digest_stream(stream)
            if not replaced:
                replacement.replace(self.archive)
                replaced = True
            return digest

        with mock.patch.object(RUNNER, "digest_stream", side_effect=swap_after_first_digest):
            _, _, hashes = RUNNER.verify_archive(self.download, self.source)
        self.assertTrue(replaced)
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

    def test_rejects_unknown_or_mistyped_entries_before_reading_members(self):
        for name, kind, message in (
            ("extra", tarfile.REGTYPE, "unsafe or duplicate archive member"),
            ("metadata", tarfile.REGTYPE, "archive member type or size differs"),
            ("metadata/pretrain.json", tarfile.DIRTYPE, "archive member type or size differs"),
        ):
            with self.subTest(name=name, kind=kind):
                extra = tarfile.TarInfo("cli3-binary-artifact/" + name)
                extra.type = kind
                extra.size = int(kind == tarfile.REGTYPE)
                self.write_archive(extra=extra)
                with mock.patch.object(tarfile.TarFile, "extractfile") as extract:
                    with self.assertRaisesRegex(ValueError, message):
                        RUNNER.verify_archive(self.download, self.source)
                    extract.assert_not_called()

    def test_rejects_special_headers_for_known_entry(self):
        for kind in (tarfile.CONTTYPE, tarfile.AREGTYPE, tarfile.GNUTYPE_SPARSE,
                     tarfile.XHDTYPE, tarfile.XGLTYPE):
            with self.subTest(kind=kind):
                extra = tarfile.TarInfo("cli3-binary-artifact/metadata/pretrain.json")
                extra.type = kind
                extra.size = 1
                self.write_archive(extra=extra)
                with mock.patch.object(tarfile.TarFile, "extractfile") as extract:
                    with self.assertRaisesRegex(ValueError, "unsafe or duplicate archive member"):
                        RUNNER.verify_archive(self.download, self.source)
                    extract.assert_not_called()

    def test_rejects_oversized_metadata_before_reading_it(self):
        extra = tarfile.TarInfo("cli3-binary-artifact/metadata/pretrain.json")
        extra.size = RUNNER.MAX_METADATA_BYTES + 1
        self.write_archive(extra=extra)
        with mock.patch.object(tarfile.TarFile, "extractfile") as extract:
            with self.assertRaisesRegex(ValueError, "archive member exceeds size limit"):
                RUNNER.verify_archive(self.download, self.source)
            extract.assert_not_called()

    def test_rejects_oversized_outer_receipt(self):
        self.write_archive()
        (self.download / (self.archive.name + ".sha256")).write_bytes(
            b"x" * (RUNNER.MAX_OUTER_SEAL_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "outer SHA-256 receipt exceeds size limit"):
            RUNNER.verify_archive(self.download, self.source)

    def test_rejects_dot_dot_archive_root(self):
        self.write_archive()
        self.archive.rename(self.download / "...tar")
        (self.download / (self.archive.name + ".sha256")).rename(self.download / "...tar.sha256")
        with self.assertRaisesRegex(ValueError, "archive root name is invalid"):
            RUNNER.verify_archive(self.download, self.source)


if __name__ == "__main__":
    unittest.main()
