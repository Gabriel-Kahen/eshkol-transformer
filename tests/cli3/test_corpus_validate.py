"""Command-level contract for the Eshkol-native D1 corpus validator."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class CorpusValidateCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.command = Path(os.environ["CLI3_CORPUS_VALIDATE_BINARY"])
        cls.writer = Path(os.environ["D1_CORPUS_TOOL"])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cli3-corpus-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "corpus"
        self.directory.mkdir()

    def run_command(self, *arguments: object, status: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [str(self.command), *(str(argument) for argument in arguments)],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
            env={"PATH": "/definitely-not-a-real-path"},
        )
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        return result

    def write_corpus(self, scenario: str = "standard") -> None:
        result = subprocess.run(
            [str(self.writer), "write", str(self.directory), scenario],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
            env={"PATH": "/definitely-not-a-real-path"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_data_error(self, category: str, *options: str) -> None:
        result = self.run_command("corpus", "validate", self.directory, *options, status=2)
        self.assertEqual(result.stdout, "")
        self.assertTrue(
            result.stderr.startswith(
                f"error: category={category} operation=token-corpus-validate message="
            ),
            result.stderr,
        )
        self.assertEqual(result.stderr.count("\n"), 1)

    def test_valid_summary(self) -> None:
        self.write_corpus()
        result = self.run_command("corpus", "validate", self.directory)
        self.assertEqual(result.stderr, "")
        self.assertEqual(
            result.stdout,
            "shards=3 tokens=7 vocab=257 limit=3 bytes=467 "
            "fingerprint=opaque-tokenizer-identity\n",
        )
        bounded = self.run_command(
            "corpus", "validate", self.directory,
            "--max-total-tokens", "7",
            "--max-shard-bytes", "161",
            "--max-manifest-bytes", "321",
        )
        self.assertEqual(bounded.stdout, result.stdout)
        self.assertEqual(bounded.stderr, "")

    def test_zero_total_limit_accepts_empty_corpus(self) -> None:
        self.write_corpus("empty")
        result = self.run_command(
            "corpus", "validate", self.directory, "--max-total-tokens", "0"
        )
        self.assertEqual(result.stderr, "")
        self.assertIn("shards=0 tokens=0", result.stdout)

    def test_missing_manifest(self) -> None:
        self.assert_data_error("io")

    def test_corrupt_shard(self) -> None:
        self.write_corpus()
        shard = self.directory / "shard-0000000000000000.ets"
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(data)
        self.assert_data_error("corrupt-data")

    def test_exceeded_limits(self) -> None:
        self.write_corpus()
        for option, value in (
            ("--max-manifest-bytes", "16"),
            ("--max-shard-bytes", "16"),
            ("--max-total-tokens", "1"),
        ):
            with self.subTest(option=option):
                self.assert_data_error("corrupt-data", option, value)

    def test_usage_errors(self) -> None:
        self.write_corpus()
        for arguments in (
            (),
            ("corpus", "validate"),
            ("corpus", "validate", str(self.directory), "--bogus", "1"),
            ("corpus", "validate", str(self.directory), "--max-shard-bytes"),
            ("corpus", "validate", str(self.directory), "--max-shard-bytes", "0"),
            ("corpus", "validate", str(self.directory), "--max-total-tokens", "-1"),
            ("corpus", "validate", str(self.directory), "--max-total-tokens", "9223372036854775808"),
            ("corpus", "validate", str(self.directory), "--max-total-tokens", "1.0"),
            ("corpus", "validate", str(self.directory), "--max-shard-bytes", "1",
             "--max-shard-bytes", "2"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_command(*arguments, status=64)
                self.assertEqual(result.stdout, "")
                self.assertIn("usage: eshkol-transformer corpus validate DIR", result.stderr)


if __name__ == "__main__":
    unittest.main()
