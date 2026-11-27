"""Installed DATA4 command and non-atomic sidecar contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from struct import unpack
import subprocess
import tempfile
import unittest


class InstalledData4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = Path(os.environ["CLI3_DATA4_BINARY"])
        cls.tokenizer = Path(os.environ["CLI3_DATA4_TOKENIZER"])
        cls.fault_shim = Path(os.environ["CLI3_DATA4_FAULT_SHIM"])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="cli3-data4-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "corpus"
        self.output.mkdir()
        for name, data in (("a", b"AB"), ("dup", b"AB"), ("b", b"CD"),
                           ("short", b"E")):
            (self.root / name).write_bytes(data)

    def command(self, *args: str, status: int = 0,
                fault: str | None = None) -> subprocess.CompletedProcess[str]:
        env = {"PATH": "/definitely-not-a-real-path"}
        if fault is not None:
            env["LD_PRELOAD"] = str(self.fault_shim)
            if fault == "manifest-read":
                env["CLI3_TEST_MANIFEST_READ_FAIL"] = "1"
            else:
                env["CLI3_TEST_FSYNC_MODE"] = fault
        result = subprocess.run(
            [str(self.binary), *map(str, args)], capture_output=True, text=True,
            timeout=30, check=False, env=env,
        )
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        return result

    def prepare(self, directory: Path | None = None, *, fault: str | None = None,
                status: int = 0, tokenizer: Path | None = None,
                sources: tuple[str, ...] = ("a", "dup", "b", "short")) -> subprocess.CompletedProcess[str]:
        directory = directory or self.output
        arguments = ["corpus", "prepare", "--tokenizer", str(tokenizer or self.tokenizer)]
        for name in sources:
            arguments.extend(("--source", f"{name}={self.root / name}"))
        arguments.extend(("--minimum-bytes", "2", "--maximum-bytes", "2",
                          "--output-directory", str(directory),
                          "--shard-token-limit", "3"))
        return self.command(*arguments, status=13 if fault else status, fault=fault)

    def validate(self, directory: Path | None = None, *, status: int = 0) -> subprocess.CompletedProcess[str]:
        return self.command("corpus", "report-validate", "--input-directory",
                            str(directory or self.output), status=status)

    def assert_report_rejected(self, category: str, status: int,
                               directory: Path | None = None) -> None:
        result = self.validate(directory, status=status)
        self.assertEqual(result.stdout, "")
        self.assertIn(f'category="{category}"', result.stderr)

    def test_installed_help_routes(self) -> None:
        group = self.command("corpus", "--help")
        self.assertIn("prepare  Filter and deduplicate", group.stdout)
        self.assertIn("report-validate  Validate a DATA4 report", group.stdout)
        self.assertEqual(group.stderr, "")
        for command in ("prepare", "report-validate"):
            help_result = self.command("corpus", command, "--help")
            self.assertIn(f"eshkol-transformer corpus {command}", help_result.stdout)
            self.assertEqual(help_result.stderr, "")

    def test_prepared_report_and_d1_summary(self) -> None:
        built = self.prepare()
        summary = json.loads(built.stdout)
        self.assertEqual(summary["total_tokens"], 4)
        self.assertEqual(built.stderr, "")
        report = (self.output / "data4-report.tsv").read_bytes()
        prefix, checksum = report.rsplit(b"checksum\t", 1)
        self.assertEqual(checksum, hashlib.sha256(prefix).hexdigest().encode() + b"\n")
        lines = prefix.splitlines()
        digest = hashlib.sha256((self.output / "manifest.etm").read_bytes())
        self.assertEqual(lines[1], b"version\t2\t0")
        self.assertEqual(lines[14:16], [
            b"total-tokens\t4", b"manifest-sha256\t" + digest.hexdigest().encode(),
        ])
        self.assertIn(b"document\t1\tdup\t2\t", report)
        self.assertIn(b"\tduplicate\ta\n", report)
        self.assertIn(b"document\t3\tshort\t1\t", report)
        self.assertIn(b"\tbelow-min\t\n", report)
        checked = self.validate()
        self.assertEqual(checked.stdout, built.stdout)
        self.assertEqual(checked.stderr, "")

    def test_report_tamper_and_malformed_row(self) -> None:
        self.prepare()
        path = self.output / "data4-report.tsv"
        original = path.read_bytes()
        altered = bytearray(original)
        altered[-2] = ord("1") if altered[-2] == ord("0") else ord("0")
        path.write_bytes(altered)
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)
        prefix = original.rsplit(b"checksum\t", 1)[0]
        malformed = prefix.replace(b"\tduplicate\ta\n", b"\tkeep\t\n", 1)
        self.assertNotEqual(malformed, prefix)
        path.write_bytes(malformed + b"checksum\t" +
                         hashlib.sha256(malformed).hexdigest().encode() + b"\n")
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)

    def test_mismatched_summary_swap_is_rejected(self) -> None:
        self.prepare()
        report = (self.output / "data4-report.tsv").read_bytes()
        other = self.root / "other"
        other.mkdir()
        (self.root / "different").write_bytes(b"XYZ")
        self.command("corpus", "prepare", "--tokenizer", str(self.tokenizer),
                     "--source", f"different={self.root / 'different'}",
                     "--minimum-bytes", "0", "--maximum-bytes", "3",
                     "--output-directory", str(other), "--shard-token-limit", "3")
        (other / "data4-report.tsv").write_bytes(report)
        self.assertIn("category=\"corrupt-data\"",
                      self.validate(other, status=12).stderr)

    def test_same_summary_swap_is_rejected(self) -> None:
        first = self.prepare()
        report = (self.output / "data4-report.tsv").read_bytes()
        other = self.root / "same-summary"
        other.mkdir()
        (self.root / "different-a").write_bytes(b"WX")
        (self.root / "different-b").write_bytes(b"YZ")
        second = self.prepare(other, sources=("different-a", "different-b"))
        self.assertEqual(json.loads(first.stdout), json.loads(second.stdout))
        self.assertEqual(self.validate(other).stdout, second.stdout)
        self.assertNotEqual((self.output / "manifest.etm").read_bytes(),
                            (other / "manifest.etm").read_bytes())
        (other / "data4-report.tsv").write_bytes(report)
        self.assert_report_rejected("corrupt-data", 12, other)

    def test_raw_bpe_report_binds_manifest_and_tokenizer(self) -> None:
        other = self.root / "other-bpe"
        other.mkdir()
        fingerprints, manifest_digests = [], []
        for training, directory, tokens in (
            (b"ABAB", self.output, (256, 67, 68)),
            (b"CDCD", other, (65, 66, 256)),
        ):
            document = self.root / f"{directory.name}.txt"
            document.write_bytes(training)
            tokenizer = document.with_suffix(".tsv")
            trained = self.command(
                "tokenizer", "train-bpe", "--document", document,
                "--maximum-merges", "1", "--minimum-frequency", "2",
                "--utf8-policy", "raw", "--output", tokenizer,
            )
            identity = json.loads(trained.stdout)
            fingerprint = identity["fingerprint"]
            self.assertEqual(identity["family"], "byte-bpe")
            self.assertEqual(identity["vocabulary_size"], 257)
            self.assertRegex(fingerprint, r"^sha256:eshkol-bpe-tokenizer-v1:[0-9a-f]{64}$")
            fingerprints.append(fingerprint)

            built = self.prepare(directory, tokenizer=tokenizer)
            self.assertEqual(json.loads(built.stdout), {
                "artifact": "token-corpus", "shard_count": 1, "shard_token_limit": 3,
                "tokenizer_fingerprint": fingerprint, "total_shard_bytes": 231,
                "total_tokens": 3, "vocabulary_size": 257,
            })
            manifest = (directory / "manifest.etm").read_bytes()
            header_bytes = int.from_bytes(manifest[12:16], "little")
            self.assertEqual(manifest[96:header_bytes], fingerprint.encode("ascii"))
            shard = (directory / "shard-0000000000000000.ets").read_bytes()
            header_bytes = int.from_bytes(shard[12:16], "little")
            self.assertEqual(unpack("<3q", shard[header_bytes:-32]), tokens)

            digest = hashlib.sha256(manifest).hexdigest().encode("ascii")
            manifest_digests.append(digest)
            report = (directory / "data4-report.tsv").read_bytes()
            self.assertEqual(report.splitlines()[14:16], [
                b"total-tokens\t3", b"manifest-sha256\t" + digest,
            ])
            checked = self.validate(directory)
            self.assertEqual(checked.stdout, built.stdout)
            self.assertEqual(checked.stderr, "")

        self.assertNotEqual(*fingerprints)
        self.assertNotEqual(*manifest_digests)
        report = (self.output / "data4-report.tsv").read_bytes()
        swapped = other / "data4-report.tsv"
        swapped.write_bytes(report)
        self.assert_report_rejected("corrupt-data", 12, other)

        # Rebind the report digest while preserving the stale tokenizer fingerprint.
        prefix = report.rsplit(b"checksum\t", 1)[0]
        prefix = prefix.replace(
            b"manifest-sha256\t" + manifest_digests[0],
            b"manifest-sha256\t" + manifest_digests[1],
        )
        swapped.write_bytes(prefix + b"checksum\t" +
                            hashlib.sha256(prefix).hexdigest().encode("ascii") + b"\n")
        self.assert_report_rejected("corrupt-data", 12, other)

    def test_legacy_unbound_report_is_rejected(self) -> None:
        self.prepare()
        path = self.output / "data4-report.tsv"
        prefix = path.read_bytes().rsplit(b"checksum\t", 1)[0]
        legacy = b"".join(
            line.replace(b"version\t2\t0", b"version\t1\t0")
            for line in prefix.splitlines(keepends=True)
            if not line.startswith(b"manifest-sha256\t")
        )
        path.write_bytes(legacy + b"checksum\t" +
                         hashlib.sha256(legacy).hexdigest().encode() + b"\n")
        self.assert_report_rejected("version-mismatch", 11)

    def test_rechecksummed_bad_manifest_fields_are_rejected(self) -> None:
        self.prepare()
        path = self.output / "data4-report.tsv"
        lines = path.read_bytes().rsplit(b"checksum\t", 1)[0].splitlines(keepends=True)
        key, digest = lines[15].rstrip(b"\n").split(b"\t")
        self.assertEqual(key, b"manifest-sha256")
        for name, value in (
            ("missing", None), ("uppercase", digest.upper()),
            ("short", digest[:-1]), ("nonhex", b"z" + digest[1:]),
            ("wrong", (b"1" if digest.startswith(b"0") else b"0") + digest[1:]),
        ):
            with self.subTest(name=name):
                field = b"" if value is None else key + b"\t" + value + b"\n"
                prefix = b"".join([*lines[:15], field, *lines[16:]])
                path.write_bytes(prefix + b"checksum\t" +
                                 hashlib.sha256(prefix).hexdigest().encode() + b"\n")
                self.assert_report_rejected("corrupt-data", 12)

    def test_missing_oversize_and_noncanonical_report(self) -> None:
        self.prepare()
        path = self.output / "data4-report.tsv"
        original = path.read_bytes()
        path.unlink()
        self.assertIn("category=\"io\"", self.validate(status=13).stderr)
        path.write_bytes(b"X" * 32769)
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)
        prefix = original.rsplit(b"checksum\t", 1)[0]
        changed = prefix.replace(b"input-count\t4\n", b"input-count\t04\n", 1)
        path.write_bytes(changed + b"checksum\t" +
                         hashlib.sha256(changed).hexdigest().encode() + b"\n")
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)

    def test_precommit_report_failure_preserves_published_d1(self) -> None:
        failed = self.prepare(fault="regular-once")
        self.assertEqual(failed.stdout, "")
        self.assertIn("corpus published; report not published by this invocation",
                      failed.stderr)
        self.assertIn('"corpus-published?":true', failed.stderr)
        self.assertIn('"report-published?":false', failed.stderr)
        self.assertTrue((self.output / "manifest.etm").is_file())
        self.assertFalse((self.output / "data4-report.tsv").exists())

    def test_manifest_read_failure_preserves_published_d1(self) -> None:
        failed = self.prepare(fault="manifest-read")
        self.assertEqual(failed.stdout, "")
        self.assertIn('category="io"', failed.stderr)
        self.assertIn("corpus published; report not published by this invocation",
                      failed.stderr)
        self.assertIn('"corpus-published?":true', failed.stderr)
        self.assertIn('"report-published?":false', failed.stderr)
        self.assertIn("checked file read failed", failed.stderr)
        manifest = self.output / "manifest.etm"
        self.assertTrue(manifest.is_file())
        self.assertFalse((self.output / "data4-report.tsv").exists())
        checked = self.command("corpus", "inspect", "--input-directory", self.output)
        self.assertEqual(json.loads(checked.stdout)["total_tokens"], 4)

    def test_dangling_sidecar_symlink_is_never_replaced(self) -> None:
        sidecar = self.output / "data4-report.tsv"
        sidecar.symlink_to(self.root / "missing-target")
        failed = self.prepare(status=13)
        self.assertEqual(failed.stdout, "")
        self.assertTrue(sidecar.is_symlink())
        self.assertFalse(sidecar.exists())
        if (self.output / "manifest.etm").exists():
            self.assertIn("corpus published; report not published by this invocation",
                          failed.stderr)
            self.assertIn('"report-published?":false', failed.stderr)
        else:
            self.assertIn("report sidecar already exists", failed.stderr)

    def test_postcommit_report_failure_marks_unknown_durability(self) -> None:
        failed = self.prepare(fault="directory-once")
        self.assertEqual(failed.stdout, "")
        self.assertIn("corpus published; report visible with uncertain durability",
                      failed.stderr)
        self.assertIn('"corpus-published?":true', failed.stderr)
        self.assertIn('"report-published?":true', failed.stderr)
        self.assertIn('"report-durability":"unknown"', failed.stderr)
        self.assertTrue((self.output / "manifest.etm").is_file())
        self.assertTrue((self.output / "data4-report.tsv").is_file())

    def test_invalid_sources_and_existing_sidecar(self) -> None:
        for source in ("bad/name=" + str(self.root / "a"), "missing-equals"):
            result = self.command(
                "corpus", "prepare", "--tokenizer", str(self.tokenizer),
                "--source", source, "--minimum-bytes", "0",
                "--maximum-bytes", "2", "--output-directory", self.output,
                "--shard-token-limit", "3", status=2,
            )
            self.assertEqual(result.stdout, "")
        self.assertFalse((self.output / "manifest.etm").exists())
        duplicated = self.command(
            "corpus", "prepare", "--tokenizer", str(self.tokenizer),
            "--source", f"x={self.root / 'a'}", "--source", f"x={self.root / 'b'}",
            "--minimum-bytes", "0", "--maximum-bytes", "2",
            "--output-directory", self.output, "--shard-token-limit", "3",
            status=10,
        )
        self.assertEqual(duplicated.stdout, "")
        self.assertFalse((self.output / "manifest.etm").exists())
        (self.output / "data4-report.tsv").write_bytes(b"occupied")
        result = self.prepare(status=13)
        self.assertEqual(result.returncode, 13)
        self.assertFalse((self.output / "manifest.etm").exists())


if __name__ == "__main__":
    unittest.main()
