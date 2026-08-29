from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from reference_format import (
    MANIFEST_FIXED,
    RECORD_BYTES,
    SHARD_FIXED,
    parse_corpus,
    put_u16,
    put_u32,
    put_u64,
    resign,
    u32,
)


class CorpusFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw_tool = os.environ.get("D1_CORPUS_TOOL")
        if not raw_tool:
            raise RuntimeError("D1_CORPUS_TOOL must name the compiled Eshkol tool")
        cls.tool = Path(raw_tool).resolve()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="d1-format-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_tool(self, *arguments: object, expected: int = 0) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [str(self.tool), *(str(argument) for argument in arguments)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env={"PATH": "/definitely-not-a-real-path"},
        )
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        return completed

    def write(self, scenario: str = "standard", name: str = "corpus") -> Path:
        directory = self.root / name
        directory.mkdir()
        self.run_tool("write", directory, scenario)
        return directory

    def assert_category(self, directory: Path, category: str, command: str = "validate") -> None:
        result = self.run_tool(command, directory, expected=2)
        self.assertIn(f"ERROR category={category}", result.stdout)

    def clone(self, source: Path, name: str) -> Path:
        destination = self.root / name
        shutil.copytree(source, destination)
        return destination

    def test_reference_round_trip_and_golden_hashes(self) -> None:
        directory = self.write()
        parsed = parse_corpus(directory)
        self.assertEqual(parsed["tokens"], [0, 1, 256, 42, 7, 8, 9])
        self.assertEqual(parsed["fingerprint"], "opaque-tokenizer-identity")
        self.assertEqual(parsed["vocab_size"], 257)
        self.assertEqual(parsed["shard_limit"], 3)
        self.assertEqual(parsed["shard_count"], 3)
        names = sorted(path.name for path in directory.iterdir())
        self.assertEqual(
            names,
            [
                "manifest.etm",
                "shard-0000000000000000.ets",
                "shard-0000000000000001.ets",
                "shard-0000000000000002.ets",
            ],
        )
        digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}
        self.assertEqual(
            digests,
            {
                "manifest.etm": "20f57deafbe49783268fd6e828b0fa76e5c868632b2247f442d7b5c9d8e35f6b",
                "shard-0000000000000000.ets": "7173f22db83ebfb9ff823671361ee3302aaefe52c0f95821c8c112c2f3c8f0ce",
                "shard-0000000000000001.ets": "5b1da5d467c5f168650f36602ed8adbea00d025f1f17e1c71660b7e9865cb377",
                "shard-0000000000000002.ets": "20f45f18c2a1c5ad0e4a7d87e34a51f424acaf5d1295b491dd14d297ddfe9ecb",
            },
        )
        self.run_tool("validate", directory)

    def test_fresh_process_determinism(self) -> None:
        first = self.write(name="first")
        second = self.write(name="second")
        first_files = sorted(path.name for path in first.iterdir())
        second_files = sorted(path.name for path in second.iterdir())
        self.assertEqual(first_files, second_files)
        for name in first_files:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_empty_and_partition_boundaries(self) -> None:
        cases = {
            "empty": (0, []),
            "limit-minus-one": (1, [0, 7]),
            "exact": (1, [0, 7, 3]),
            "limit-plus-one": (2, [0, 1, 2, 3]),
            "two-limit": (2, [0, 1, 2, 3, 4, 5]),
            "two-limit-plus-one": (3, [0, 1, 2, 3, 4, 5, 6]),
            "vocab-one": (1, [0]),
            "max-token": (1, [9223372036854775806]),
        }
        for scenario, (shards, tokens) in cases.items():
            with self.subTest(scenario=scenario):
                parsed = parse_corpus(self.write(scenario, scenario))
                self.assertEqual(parsed["shard_count"], shards)
                self.assertEqual(parsed["tokens"], tokens)

        fingerprint_max = parse_corpus(self.write("fingerprint-max", "fingerprint-max"))
        self.assertEqual(fingerprint_max["fingerprint"], "a" * 192)
        fingerprint_utf8 = parse_corpus(self.write("fingerprint-utf8", "fingerprint-utf8"))
        self.assertEqual(fingerprint_utf8["fingerprint"].encode("utf-8"),
                         "opaque-λ-identity".encode("utf-8"))

    def test_invalid_writer_arguments_leave_no_manifest(self) -> None:
        for scenario in (
            "invalid-negative",
            "invalid-vocab",
            "invalid-large",
            "invalid-noninteger",
            "invalid-improper",
            "invalid-cycle",
            "invalid-inexact-token",
            "invalid-vocab-size",
            "invalid-limit",
            "invalid-inexact-vocab",
            "invalid-inexact-limit",
            "invalid-fingerprint",
            "invalid-fingerprint-long",
        ):
            with self.subTest(scenario=scenario):
                directory = self.root / scenario
                directory.mkdir()
                result = self.run_tool("write", directory, scenario, expected=2)
                self.assertIn("ERROR category=invalid-argument", result.stdout)
                self.assertFalse((directory / "manifest.etm").exists())
                self.assertFalse((directory / ".d1-writer-lock").exists())

    def test_preexisting_target_and_writer_lock_are_rejected(self) -> None:
        corpus = self.write()
        result = self.run_tool("write", corpus, "standard", expected=2)
        self.assertIn("ERROR category=invalid-argument", result.stdout)

        locked = self.root / "locked"
        locked.mkdir()
        (locked / ".d1-writer-lock").mkdir()
        result = self.run_tool("write", locked, "standard", expected=2)
        self.assertIn("ERROR category=invalid-argument", result.stdout)
        self.assertFalse((locked / "manifest.etm").exists())

    def test_reader_limits_are_enforced(self) -> None:
        corpus = self.write()
        self.assert_category(corpus, "corrupt-data", "validate-small-manifest")
        self.assert_category(corpus, "corrupt-data", "validate-small-shard")
        self.assert_category(corpus, "corrupt-data", "validate-small-total")
        for command in (
            "validate-invalid-manifest-limit",
            "validate-invalid-shard-limit",
            "validate-invalid-total-limit",
        ):
            self.assert_category(corpus, "invalid-argument", command)
        self.assert_category(corpus, "invalid-argument", "invalid-summary")
        forged = self.run_tool("invalid-summary-procedure", corpus, expected=2)
        self.assertIn("ERROR category=invalid-argument", forged.stdout)
        self.assertIn("operation=token-corpus-summary-shard-count", forged.stdout)

        empty = self.write("empty", "empty-zero-policy")
        self.run_tool("validate-zero-total", empty)
        self.assert_category(corpus, "corrupt-data", "validate-zero-total")

    def test_checksum_truncation_trailing_and_missing_rejection(self) -> None:
        original = self.write()
        manifest_flip = self.clone(original, "manifest-flip")
        data = bytearray((manifest_flip / "manifest.etm").read_bytes())
        data[40] ^= 1
        (manifest_flip / "manifest.etm").write_bytes(data)
        self.assert_category(manifest_flip, "corrupt-data")

        shard_flip = self.clone(original, "shard-flip")
        shard = shard_flip / "shard-0000000000000000.ets"
        data = bytearray(shard.read_bytes())
        data[-33] ^= 1
        shard.write_bytes(data)
        self.assert_category(shard_flip, "corrupt-data")

        for name, target in (("truncated", "manifest.etm"), ("truncated-shard", "shard-0000000000000000.ets")):
            corpus = self.clone(original, name)
            path = corpus / target
            path.write_bytes(path.read_bytes()[:-1])
            self.assert_category(corpus, "corrupt-data")

        trailing = self.clone(original, "trailing")
        path = trailing / "manifest.etm"
        path.write_bytes(path.read_bytes() + b"x")
        self.assert_category(trailing, "corrupt-data")

        trailing_shard = self.clone(original, "trailing-shard")
        path = trailing_shard / "shard-0000000000000000.ets"
        path.write_bytes(path.read_bytes() + b"x")
        self.assert_category(trailing_shard, "corrupt-data")

        missing = self.clone(original, "missing")
        (missing / "shard-0000000000000001.ets").unlink()
        self.assert_category(missing, "io")

    def mutate_manifest(self, original: Path, name: str, mutation) -> Path:
        corpus = self.clone(original, name)
        path = corpus / "manifest.etm"
        data = bytearray(path.read_bytes())
        mutation(data)
        resign(data)
        path.write_bytes(data)
        return corpus

    def mutate_shard(self, original: Path, name: str, mutation, index: int = 0) -> Path:
        corpus = self.clone(original, name)
        shard_path = corpus / f"shard-{index:016d}.ets"
        shard = bytearray(shard_path.read_bytes())
        mutation(shard)
        resign(shard)
        shard_path.write_bytes(shard)
        manifest_path = corpus / "manifest.etm"
        manifest = bytearray(manifest_path.read_bytes())
        record = u32(manifest, 12) + index * RECORD_BYTES
        manifest[record + 24 : record + 56] = shard[-32:]
        resign(manifest)
        manifest_path.write_bytes(manifest)
        return corpus

    def test_resigned_manifest_structure_and_version_rejection(self) -> None:
        original = self.write()
        cases = [
            ("magic", lambda data: data.__setitem__(0, ord("X")), "corrupt-data"),
            ("major", lambda data: put_u16(data, 8, 2), "version-mismatch"),
            ("minor", lambda data: put_u16(data, 10, 1), "version-mismatch"),
            ("feature", lambda data: put_u32(data, 16, 1), "unsupported"),
            ("encoding", lambda data: put_u32(data, 20, 2), "unsupported"),
            ("checksum", lambda data: put_u32(data, 24, 2), "unsupported"),
            ("reserved", lambda data: data.__setitem__(28, 1), "corrupt-data"),
            ("total", lambda data: put_u64(data, 40, 8), "corrupt-data"),
            ("total-shard-bytes", lambda data: put_u64(data, 48, 1), "corrupt-data"),
            ("header-size", lambda data: put_u32(data, 12, MANIFEST_FIXED), "corrupt-data"),
            ("fingerprint-size", lambda data: put_u32(data, 72, 0), "corrupt-data"),
            ("record-size", lambda data: put_u32(data, 76, 55), "corrupt-data"),
            ("file-size", lambda data: put_u64(data, 80, len(data) - 1), "corrupt-data"),
            ("high-shard-count", lambda data: put_u64(data, 32, 1 << 63), "corrupt-data"),
            ("high-token-count", lambda data: put_u64(data, 40, 1 << 63), "corrupt-data"),
            ("record-file-bytes", lambda data: put_u64(data, u32(data, 12) + 16, 1),
             "corrupt-data"),
        ]
        for name, mutation, category in cases:
            with self.subTest(name=name):
                self.assert_category(self.mutate_manifest(original, name, mutation), category)

        invalid_utf8 = self.mutate_manifest(
            original, "manifest-invalid-utf8",
            lambda data: data.__setitem__(MANIFEST_FIXED, 0xFF),
        )
        self.assert_category(invalid_utf8, "corrupt-data")

    def test_duplicate_record_and_validly_resigned_invalid_token(self) -> None:
        original = self.write()
        duplicate = self.clone(original, "duplicate-record")
        manifest_path = duplicate / "manifest.etm"
        manifest = bytearray(manifest_path.read_bytes())
        header = u32(manifest, 12)
        put_u64(manifest, header + RECORD_BYTES, 0)
        resign(manifest)
        manifest_path.write_bytes(manifest)
        self.assert_category(duplicate, "corrupt-data")

        invalid = self.clone(original, "invalid-token-resigned")
        shard_path = invalid / "shard-0000000000000000.ets"
        shard = bytearray(shard_path.read_bytes())
        shard_header = u32(shard, 12)
        put_u64(shard, shard_header, 257)
        resign(shard)
        shard_path.write_bytes(shard)
        manifest_path = invalid / "manifest.etm"
        manifest = bytearray(manifest_path.read_bytes())
        manifest_header = u32(manifest, 12)
        manifest[manifest_header + 24 : manifest_header + 56] = shard[-32:]
        resign(manifest)
        manifest_path.write_bytes(manifest)
        self.assert_category(invalid, "corrupt-data")

        negative = self.clone(original, "negative-token-resigned")
        shard_path = negative / "shard-0000000000000000.ets"
        shard = bytearray(shard_path.read_bytes())
        put_u64(shard, u32(shard, 12), (1 << 64) - 1)
        resign(shard)
        shard_path.write_bytes(shard)
        manifest_path = negative / "manifest.etm"
        manifest = bytearray(manifest_path.read_bytes())
        manifest_header = u32(manifest, 12)
        manifest[manifest_header + 24 : manifest_header + 56] = shard[-32:]
        resign(manifest)
        manifest_path.write_bytes(manifest)
        self.assert_category(negative, "corrupt-data")

    def test_resigned_shard_structure_version_and_algorithm_rejection(self) -> None:
        original = self.write()
        cases = [
            ("shard-magic", lambda data: data.__setitem__(0, ord("X")), "corrupt-data"),
            ("shard-major", lambda data: put_u16(data, 8, 2), "version-mismatch"),
            ("shard-minor", lambda data: put_u16(data, 10, 1), "version-mismatch"),
            ("shard-feature", lambda data: put_u32(data, 16, 1), "unsupported"),
            ("shard-encoding", lambda data: put_u32(data, 20, 2), "unsupported"),
            ("shard-checksum", lambda data: put_u32(data, 24, 2), "unsupported"),
            ("shard-reserved-a", lambda data: data.__setitem__(28, 1), "corrupt-data"),
            ("shard-reserved-b", lambda data: data.__setitem__(68, 1), "corrupt-data"),
            ("shard-index", lambda data: put_u64(data, 32, 1), "corrupt-data"),
            ("shard-count", lambda data: put_u64(data, 40, 2), "corrupt-data"),
            ("shard-payload", lambda data: put_u64(data, 48, 16), "corrupt-data"),
            ("shard-vocab", lambda data: put_u64(data, 56, 258), "corrupt-data"),
            ("shard-fingerprint-size", lambda data: put_u32(data, 64, 0), "corrupt-data"),
            ("shard-file-size", lambda data: put_u64(data, 72, len(data) - 1), "corrupt-data"),
            ("shard-high-count", lambda data: put_u64(data, 40, 1 << 63), "corrupt-data"),
            ("shard-high-payload", lambda data: put_u64(data, 48, 1 << 63), "corrupt-data"),
            ("shard-high-file-size", lambda data: put_u64(data, 72, 1 << 63), "corrupt-data"),
        ]
        for name, mutation, category in cases:
            with self.subTest(name=name):
                self.assert_category(self.mutate_shard(original, name, mutation), category)

        invalid_utf8 = self.mutate_shard(
            original, "shard-invalid-utf8",
            lambda data: data.__setitem__(SHARD_FIXED, 0xFF),
        )
        self.assert_category(invalid_utf8, "corrupt-data")

    def test_validly_resigned_tokenizer_identity_conflict(self) -> None:
        original = self.write()
        corpus = self.clone(original, "identity-conflict")
        shard_path = corpus / "shard-0000000000000000.ets"
        shard = bytearray(shard_path.read_bytes())
        shard[SHARD_FIXED] ^= 1
        resign(shard)
        shard_path.write_bytes(shard)
        manifest_path = corpus / "manifest.etm"
        manifest = bytearray(manifest_path.read_bytes())
        header = u32(manifest, 12)
        manifest[header + 24 : header + 56] = shard[-32:]
        resign(manifest)
        manifest_path.write_bytes(manifest)
        self.assert_category(corpus, "corrupt-data")

    def test_no_manifest_is_visible_when_writer_fails_before_publication(self) -> None:
        directory = self.root / "blocked"
        directory.mkdir()
        (directory / "shard-0000000000000001.ets").write_bytes(b"preexisting")
        result = self.run_tool("write", directory, "standard", expected=2)
        self.assertIn("ERROR category=invalid-argument", result.stdout)
        self.assertFalse((directory / "manifest.etm").exists())
        self.assertFalse((directory / "shard-0000000000000000.ets").exists())
        self.assertEqual((directory / "shard-0000000000000001.ets").read_bytes(), b"preexisting")
        self.assertFalse((directory / ".d1-writer-lock").exists())

        failed = self.root / "manifest-write-failure"
        failed.mkdir()
        unrelated = failed / "unrelated.txt"
        unrelated.write_bytes(b"preserve")

        injected = threading.Event()
        stop = threading.Event()

        def inject_manifest_temporary() -> None:
            first_shard = failed / "shard-0000000000000000.ets"
            manifest_temporary = failed / "manifest.etm.tmp"
            while not stop.is_set():
                if first_shard.exists():
                    try:
                        with manifest_temporary.open("xb") as stream:
                            stream.write(b"external-preexisting")
                        injected.set()
                        return
                    except FileExistsError:
                        return
                time.sleep(0.0005)

        watcher = threading.Thread(target=inject_manifest_temporary, daemon=True)
        watcher.start()
        completed = subprocess.run(
            [str(self.tool), "write", str(failed), "publication-limit"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env={"PATH": "/definitely-not-a-real-path"},
        )
        stop.set()
        watcher.join(timeout=2)
        self.assertTrue(injected.is_set(), "fault injector did not observe a shard")
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("ERROR category=invalid-argument", completed.stdout)
        manifest_temporary = failed / "manifest.etm.tmp"
        self.assertEqual(manifest_temporary.read_bytes(), b"external-preexisting")
        manifest_temporary.unlink()
        self.assertEqual(sorted(path.name for path in failed.iterdir()), ["unrelated.txt"])
        self.assertEqual(unrelated.read_bytes(), b"preserve")

    def test_unreferenced_files_are_ignored_but_never_followed(self) -> None:
        corpus = self.write()
        (corpus / "unrelated.txt").write_text("not part of the corpus", encoding="utf-8")
        (corpus / "shard-9999999999999999.ets").write_bytes(b"orphan")
        self.run_tool("validate", corpus)

    def test_filesystem_permission_failure_is_io(self) -> None:
        corpus = self.root / "foreign-file-error"
        corpus.mkdir()
        manifest = corpus / "manifest.etm"
        manifest.write_bytes(bytes(256))
        manifest.chmod(0)
        try:
            completed = self.run_tool("validate", corpus, expected=2)
        finally:
            manifest.chmod(0o600)
        self.assertIn("ERROR category=io", completed.stdout)


if __name__ == "__main__":
    unittest.main()
