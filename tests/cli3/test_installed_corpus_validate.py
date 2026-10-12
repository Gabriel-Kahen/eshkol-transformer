"""Installed CLI3 corpus validation against a corpus built by the same binary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


class InstalledCorpusValidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = Path(os.environ["CLI3_INSTALLED_BINARY"])
        cls.fixture = Path(os.environ["CLI3_INSTALLED_CORPUS"])
        cls.summary = json.loads(Path(os.environ["CLI3_INSTALLED_CORPUS_JSON"]).read_text())

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="cli3-installed-validate-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "corpus"
        shutil.copytree(self.fixture, self.directory)

    def run_command(self, *args: object, status: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [str(self.binary), *(str(arg) for arg in args)],
            capture_output=True, text=True, timeout=20, check=False,
            env={"PATH": "/definitely-not-a-real-path"},
        )
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        return result

    def assert_data_error(self, category: str, *options: str) -> None:
        result = self.run_command("corpus", "validate", self.directory, *options, status=2)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith(
            f"error: category={category} operation=token-corpus-validate message="
        ), result.stderr)
        self.assertEqual(result.stderr.count("\n"), 1)

    def test_valid_summary_and_bounded_options(self) -> None:
        expected = (
            f"shards={self.summary['shard_count']} "
            f"tokens={self.summary['total_tokens']} "
            f"vocab={self.summary['vocabulary_size']} "
            f"limit={self.summary['shard_token_limit']} "
            f"bytes={self.summary['total_shard_bytes']} "
            f"fingerprint_hex={self.summary['tokenizer_fingerprint'].encode().hex()}\n"
        )
        result = self.run_command("corpus", "validate", self.directory)
        self.assertEqual(result.stdout, expected)
        self.assertEqual(result.stderr, "")
        shards = list(self.directory.glob("shard-*.ets"))
        self.assertTrue(shards)
        bounded = self.run_command(
            "corpus", "validate", self.directory,
            "--max-total-tokens", self.summary["total_tokens"],
            "--max-shard-bytes", max(path.stat().st_size for path in shards),
            "--max-manifest-bytes", (self.directory / "manifest.etm").stat().st_size,
        )
        self.assertEqual(bounded.stdout, expected)
        self.assertEqual(bounded.stderr, "")

    def test_usage_rejects_unknown_duplicate_and_overflow(self) -> None:
        for args in (
            ("corpus", "validate"),
            ("corpus", "validate", ""),
            ("corpus", "validate", self.directory, "--bogus", "1"),
            ("corpus", "validate", self.directory, "--max-shard-bytes"),
            ("corpus", "validate", self.directory, "--max-shard-bytes", "0"),
            ("corpus", "validate", self.directory, "--max-total-tokens", "-1"),
            ("corpus", "validate", self.directory, "--max-total-tokens", "9223372036854775808"),
            ("corpus", "validate", self.directory, "--max-total-tokens", "1.0"),
            ("corpus", "validate", self.directory, "--max-shard-bytes", "1",
             "--max-shard-bytes", "2"),
        ):
            with self.subTest(args=args):
                result = self.run_command(*args, status=64)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr.count("\n"), 2)
                self.assertIn("usage: eshkol-transformer corpus validate DIR", result.stderr)

    def test_missing_manifest_corrupt_shard_and_policy_limits(self) -> None:
        (self.directory / "manifest.etm").unlink()
        self.assert_data_error("io")
        shutil.rmtree(self.directory)
        shutil.copytree(self.fixture, self.directory)
        shard = next(self.directory.glob("shard-*.ets"))
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(data)
        self.assert_data_error("corrupt-data")
        shutil.rmtree(self.directory)
        shutil.copytree(self.fixture, self.directory)
        for option, value in (("--max-manifest-bytes", "16"),
                              ("--max-shard-bytes", "16"),
                              ("--max-total-tokens", "1")):
            with self.subTest(option=option):
                self.assert_data_error("corrupt-data", option, value)

    def test_version_and_required_feature_categories(self) -> None:
        for offset, format_, value, category in (
            (8, "<H", 2, "version-mismatch"),
            (16, "<I", 1, "unsupported"),
        ):
            with self.subTest(category=category):
                manifest_path = self.directory / "manifest.etm"
                manifest = bytearray(manifest_path.read_bytes())
                struct.pack_into(format_, manifest, offset, value)
                manifest[-32:] = hashlib.sha256(manifest[:-32]).digest()
                manifest_path.write_bytes(manifest)
                self.assert_data_error(category)
                shutil.copyfile(self.fixture / "manifest.etm", manifest_path)

    def test_existing_commands_keep_their_dispatch(self) -> None:
        version = self.run_command("--version")
        self.assertEqual(version.stdout, "eshkol-transformer 0.0.1\n")
        self.assertEqual(version.stderr, "")
        inspect = self.run_command("corpus", "inspect", "--input-directory", self.directory)
        self.assertEqual(json.loads(inspect.stdout), self.summary)
        self.assertEqual(inspect.stderr, "")


if __name__ == "__main__":
    unittest.main()
